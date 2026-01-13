import os
import random
import shutil
import numpy as np
import torch

# Suppress CUDA graph dynamic shape warnings
import torch._inductor.config
torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit = None

from sys import (stdout)
from time import perf_counter_ns
from skyrimnet_chatterbox.cache_utils import (
    load_conditionals_cache,
   save_conditionals_cache,
   get_cache_key,
   save_torchaudio_wav
)
from skyrimnet_chatterbox.chatterbox.tensor_utils import initialize_model_dtype, safe_conditional_to_dtype
# Third-party imports
from pathlib import Path

from loguru import logger

#MULTILINGUAL = True
#from skyrimnet_chatterbox.chatterbox.mtl_tts import ChatterboxMultilingualTTS as Chatterbox

MULTILINGUAL = False
#from skyrimnet_chatterbox.chatterbox.tts import ChatterboxTTS as Chatterbox

from skyrimnet_chatterbox.chatterbox.tts_turbo import ChatterboxTurboTTS as Chatterbox


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
print(f"Using device: {DEVICE}, dtype: {DTYPE}")
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model():
    model = Chatterbox.from_pretrained(DEVICE)
    initialize_model_dtype(model, DTYPE)
    torch.cuda.empty_cache()
    return model


def generate(model, text, audio_prompt_path, exaggeration=0.5, temperature=1.0, seed_num=0, cfgw=0,language_id="en"):
    enable_memory_cache = True
    enable_disk_cache = True
    cache_voice = True
    device = DEVICE
    dtype = DTYPE

    if model is None:
        model = load_model()

    if seed_num != 0:
        set_seed(int(seed_num))

    func_start_time = perf_counter_ns()

    # Enhanced conditional preparation with configurable caching
    if audio_prompt_path is not None:
        # Generate cache key
        cache_key = get_cache_key(audio_prompt_path, exaggeration)
        conditionals_loaded = False
        # Try to load from cache first (respecting cache flags)
        if cache_key and (enable_memory_cache or enable_disk_cache):
            #def load_conditionals_cache(language: str, cache_key: str, model, device, dtype, enable_memory_cache=True, enable_disk_cache=True):

            if load_conditionals_cache(language=language_id, cache_key=cache_key, model=model, device=device, dtype=dtype, enable_memory_cache=enable_memory_cache, enable_disk_cache=enable_disk_cache):
                conditionals_loaded = True
        # If not loaded from cache, prepare and optionally cache
        if not conditionals_loaded:
            model.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
            if dtype != torch.float32:
                safe_conditional_to_dtype(model, dtype)
            # Save to cache if we have a cache key and caching is enabled
            if cache_key and (enable_memory_cache or enable_disk_cache):
                save_conditionals_cache(language=language_id, cache_key=cache_key, cond_cls=model.conds, enable_memory_cache=enable_memory_cache, enable_disk_cache=enable_disk_cache)
        # Update in-memory cache tracking
        if cache_voice:
            model._cached_prompt_path = audio_prompt_path
    conditional_start_time = perf_counter_ns()
    #logger.info(f"Conditionals prepared. Time: {(conditional_start_time - func_start_time) / 1_000_000_000:.2f}s")
    generate_start_time = perf_counter_ns()
    t3_params={
        # Turbo model optimization params:
        #"generate_token_backend": "cudagraphs-manual", # manual CUDA graphs (recaptures each generation)
        #"generate_token_backend": "cudagraphs",
        # "generate_token_backend": "eager", # no compilation, baseline
        # "generate_token_backend": "inductor", # requires triton
        "generate_token_backend": "reduce-overhead", # requires triton
        # "generate_token_backend": "inductor-strided",
        # "generate_token_backend": "cudagraphs-strided",
        "stride_length": 4, # "strided" options compile <1-2-3-4> iteration steps together, which improves performance by reducing memory copying issues in torch.compile
        "skip_when_1": True, # skips Top P when it's set to 1.0
        "benchmark": False, # Synchronizes CUDA to get the real it/s
        
        # Note: "initial_forward_pass_backend" is NOT supported by turbo model (only regular inference())
    }
    generate_args={
        "text": text,
        "exaggeration": exaggeration,
        "temperature": temperature,
        "cfg_weight": cfgw,
        "min_p": 0.05,
        "top_p": 1.0,
        "repetition_penalty": 1.2,
        "disable_tqdm": True,
        "t3_params": t3_params
    }

    if MULTILINGUAL:
        generate_args["language_id"] = language_id

    wav = model.generate(
        **generate_args
    )
    logger.info(f"Generation completed. Time: {(perf_counter_ns() - generate_start_time) / 1_000_000_000:.2f}s")
    # Log execution time
    func_end_time = perf_counter_ns()

    total_duration_s = (func_end_time - func_start_time)  / 1_000_000_000  # Convert nanoseconds to seconds
    wav_length = wav.shape[-1]   / model.sr

    logger.info(f"Generated audio length: {wav_length:.2f} seconds {model.sr}. Speed: {wav_length / total_duration_s:.2f}x")
    wave_file = str(save_torchaudio_wav(wav, model.sr, audio_path=audio_prompt_path))
    del wav
    torch.cuda.empty_cache()
    return wave_file
    #return (model.sr, wav.squeeze(0).cpu().numpy())


if __name__ == "__main__":
    shutil.rmtree(Path("cache").joinpath("conditionals"), ignore_errors=True)
    test_text= "Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible."
   
    test_text0 = "Kolb and the Dragon… a children's tale dressed up as heroism."
    test_text1 = " I flip the pages and every choice is a death or a cheat: trust the wrong cave, trust the wrong tavern, trust the wrong body part of the beast and you're done—bones for broth, meat for the pot. The book pretends it's about bravery, but it's really about luck and paranoia. Take the windy tunnel? Wind snuffs your torch and you break your skull. Rest in the elf-run tavern? They poison the mead."
    test_text2 = test_text0 + test_text1
    test_text3 = " Swing for the dragon's soft belly? It swallows you whole. Only the neck works, only the cold tunnel works, only the gold for the ghost works. Every other path is a corpse. I read it twice, tracking the branches like a battle map. Seventeen ways to die, one way to win, and even that victory feels thin—Kolb goes home, village cheers, dragon stops burning. No mention of the scent that must've clung to his clothes after he sawed through scale and sinew, no mention of the nightmares when he shuts his eyes and sees the lair floor carpeted with picked-clean ribs. The story stops before the real cost comes due."
    test_text4 = " Reminds me of every “simple” job we take. Get the girl, burn the ledgers, kill the slaver—clean, heroic, done. But there's always a windy tunnel we didn't scout, always a smiling elf pouring the mead. Last week in Falkreath the jailor's “broken lock” looked like the safe path until I tasted the drugged wine on the air. We pulled Sanbri out, but the ghost of that place is still clinging to my tongue."
    test_text5 = " I keep the book open to page sixteen: dragon asleep, throat and belly offered like choices. I've struck both in real life—neck for the quick kill, belly for the message. Neither ends the story; it just buys you a breath before the next beast wakes. Maybe that's the real lesson Kolb's too young to learn: winning isn't surviving, it's deciding which death you can carry."
    test_text6 = test_text2 + test_text3 + test_text4 + test_text5
   
    test_asset2=Path.cwd().joinpath("assets", "dlc1seranavoice.wav")
    test_asset = Path.cwd().joinpath("assets", "fishaudio_horror.wav")
    model = load_model()
    #wavfile = generate(model, "[happy] " + test_text0, language_id="en", audio_prompt_path=test_asset2, exaggeration=0.55, temperature=1.0, seed_num=420, cfgw=0.35)
    #wavfile = generate(model, "[happy] Hello!", language_id="en", audio_prompt_path=test_asset2, exaggeration=0.55, temperature=1.0, seed_num=420, cfgw=0.35)
    #wavfile = generate(model, "[happy] " + test_text2, language_id="en", audio_prompt_path=test_asset2, exaggeration=0.55, temperature=1.0, seed_num=420, cfgw=0.35)
    #wavfile = generate(model, "[happy] " + test_text3, language_id="en", audio_prompt_path=test_asset2, exaggeration=0.55, temperature=1.0, seed_num=420, cfgw=0.35)
    wavfile = generate(model, "[happy] " + test_text6, language_id="en", audio_prompt_path=test_asset2, seed_num=420)
    #wavfile = generate(model, "[angry] " + test_text, language_id="en", audio_prompt_path=test_asset, exaggeration=0.55, temperature=1.0, seed_num=420, cfgw=0.35)
    #wavfile = generate(model, "[whispering] " + test_text, language_id="en", audio_prompt_path=test_asset2, exaggeration=0.55, temperature=1.0, seed_num=420, cfgw=0.35)
    #wavfile = generate(model, "[sarcastic] " + test_text, language_id="en", audio_prompt_path=test_asset, exaggeration=0.55, temperature=1.0, seed_num=420, cfgw=0.35)
    #wavfile = generate(model, "[happy] " + test_text, language_id="en", audio_prompt_path=test_asset, exaggeration=0.55, temperature=1.0, seed_num=420, cfgw=0.35)
    #wavfile = generate(model, "[angry] " + test_text, language_id="en", audio_prompt_path=test_asset2, exaggeration=0.55, temperature=1.0, seed_num=420, cfgw=0.35)
    wavfile = generate(model, "A short story about a dragon and a hero.", language_id="en", audio_prompt_path=test_asset, seed_num=420)
    wavfile = generate(model, "[sarcastic] " + test_text6, language_id="en", audio_prompt_path=test_asset, seed_num=420)
    
    #wavfile = generate(model, "[sarcastic] " + test_text, language_id="en", audio_prompt_path=test_asset2, exaggeration=0.55, temperature=1.0, seed_num=420, cfgw=0.35)

    #wavfile = generate(model, "你好世界", language_id="zh", audio_prompt_path=test_asset2, exaggeration=0.65, temperature=0.8, seed_num=420, cfgw=0)
    print(f"Generated wav file: {wavfile}")

