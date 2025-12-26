from pathlib import Path

from huggingface_hub import hf_hub_download

import torch

from .conditionals import Conditionals
from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond
from .tensor_utils import (
    load_t3_state_dict_safe,
    load_s3gen_safe, 
    load_voice_encoder_safe,
    load_conditionals_safe
)
from .shared_utils import (
    check_mps_availability,
    validate_audio_file,
    drop_bad_tokens,
    prepare_text_tokens,
    punc_norm,
    validate_exaggeration,
    check_exaggeration_update_needed,
    validate_text_input,
    validate_float_parameter,
    validate_audio_prompt_path,
    smart_text_splitter,
    estimate_token_count,
    concatenate_audio_tensors
)
from .shared_audio_utils import load_and_preprocess_audio


REPO_ID = "ResembleAI/chatterbox"


class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds

    @classmethod
    def from_local(cls, ckpt_dir, s3_ckpt_dir, device) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)

        # Load voice encoder
        ve = load_voice_encoder_safe(ckpt_dir, device, is_multilingual=False)

        # Load T3 model
        t3 = T3()
        load_t3_state_dict_safe(t3, ckpt_dir / "t3_cfg.safetensors", device)

        # Load S3Gen model
        s3gen = load_s3gen_safe(s3_ckpt_dir, device, is_multilingual=False)

        # Load tokenizer
        tokenizer = EnTokenizer(str(ckpt_dir / "tokenizer.json"))

        # Load conditionals if they exist
        conds = load_conditionals_safe(ckpt_dir, device, is_multilingual=False)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTTS':
        # Use shared MPS checking utility
        device = check_mps_availability(device)

        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors","tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath,  cache_dir="models",)
            
        s3_local_path = None
        for fpath in ["s3gen.safetensors","s3gen_meanflow.safetensors"]:
            s3_local_path = hf_hub_download(repo_id=f"{REPO_ID}-turbo", filename=fpath,  cache_dir="models",)
        s3_path = s3_local_path if s3_local_path else local_path
        return cls.from_local(ckpt_dir=Path(local_path).parent, s3_ckpt_dir=Path(s3_path).parent, device=device)
    
    def set_conditionals(self, conds: Conditionals):
        """
        Set the conditionals for T3 and S3Gen.
        """
        if conds is not None:
            self.conds = conds.to(self.device)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        # Validate inputs
        validate_audio_file(wav_fpath)
        exaggeration = validate_exaggeration(exaggeration)
        
        # Use shared audio preprocessing
        s3gen_ref_wav, ref_16k_wav_tensor = load_and_preprocess_audio(wav_fpath, self.device)

        # Slice as tensor
        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)
        
        # Speech cond prompt tokens
        t3_cond_prompt_tokens = None
        plen = getattr(self.t3.hp, 'speech_cond_prompt_len', 30)
        if plen:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav_tensor[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding - use tensor-native method, keep on device
        ve_embed = self.ve.embeds_from_wavs_tensor([ref_16k_wav_tensor], sample_rate=S3_SR)
        ve_embed = ve_embed.mean(dim=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1, dtype=ve_embed.dtype),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        language_id=None,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        # stream - left for API compatibility
        tokens_per_slice=None,
        remove_milliseconds=None,
        remove_milliseconds_start=None,
        chunk_overlap_method=None,
        # cache optimization params
        max_new_tokens=750, 
        max_cache_len=1050, # Affects the T3 speed, hence important
        # t3 sampling params
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        disable_tqdm=False,
        t3_params={},
    ):
        # Validate inputs using shared utilities
        text = validate_text_input(text)
        exaggeration = validate_float_parameter(exaggeration, "exaggeration", 0.0, 2.0)
        cfg_weight = validate_float_parameter(cfg_weight, "cfg_weight", 0.0, 1.0)
        temperature = validate_float_parameter(temperature, "temperature", allow_zero=False)
        min_p = validate_float_parameter(min_p, "min_p", 0.0, 1.0)
        top_p = validate_float_parameter(top_p, "top_p", 0.0, 1.0)
        repetition_penalty = validate_float_parameter(repetition_penalty, "repetition_penalty", allow_zero=False)

        if tokens_per_slice is not None or remove_milliseconds is not None or remove_milliseconds_start is not None or chunk_overlap_method is not None:
            print("Streaming by token slices has been discontinued due to audio clipping. Continuing with full generation.")

        if audio_prompt_path:
            validate_audio_prompt_path(audio_prompt_path)
            print(f"cond size before preparing: {self.conds}")
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed using shared utility
        if self.conds is not None:
            needs_update, new_emotion_tensor = check_exaggeration_update_needed(
                self.conds.t3.emotion_adv, exaggeration, self.device
            )
            if needs_update:
                _cond: T3Cond = self.conds.t3
                self.conds.t3 = T3Cond(
                    speaker_emb=_cond.speaker_emb,
                    cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                    emotion_adv=new_emotion_tensor,
                ).to(device=self.device, dtype=self.conds.t3.speaker_emb.dtype)

        # Normalize and check if text needs chunking
        text = punc_norm(text)
        
        # Check if text needs to be chunked based on token count
        estimated_tokens = estimate_token_count(text, self.tokenizer)
        #print(f"Estimated token count: {estimated_tokens}")
        # Set chunk limit based on cache constraints: max_cache_len - max_new_tokens
        # This prevents the "max_cache_len too small" warning and ensures optimal performance
        max_chunk_tokens = max(200, max_cache_len - max_new_tokens - 75)  # 75 token safety margin
        #print(f"Using max chunk token limit: {max_chunk_tokens}")
        if estimated_tokens <= max_chunk_tokens:
            # Text is small enough - process normally without chunking
            return self._generate_single_chunk(
                text, cfg_weight, max_new_tokens, temperature, 
                max_cache_len, repetition_penalty, min_p, top_p, disable_tqdm=disable_tqdm, t3_params=t3_params
            )
        else:
            # Text is too large - split into chunks and process separately
            #print(f"Text too large ({estimated_tokens} tokens), splitting into chunks...")
            #print(f"Using cache-aware chunk limit: {max_chunk_tokens} tokens (cache_len={max_cache_len}, max_new={max_new_tokens})")
            text_chunks = smart_text_splitter(text, max_chunk_tokens, self.tokenizer)
            
            # Store original conditionals for reuse
            original_conds = self.conds.clone()
            
            audio_chunks = []
            for i, chunk in enumerate(text_chunks):
                #print(f"Processing chunk {i+1}/{len(text_chunks)}")
                #print(f"Chunk : {chunk}")
                # Reset conditionals for each chunk to ensure consistency
                self.conds = original_conds.clone()
                
                chunk_audio = self._generate_single_chunk(
                    chunk, cfg_weight, max_new_tokens, temperature,
                    max_cache_len, repetition_penalty, min_p, top_p, disable_tqdm=disable_tqdm, t3_params=t3_params
                )
                audio_chunks.append(chunk_audio)
            
            # Concatenate all audio chunks with brief silence between them
            return concatenate_audio_tensors(audio_chunks, silence_duration=0.1, sample_rate=self.sr)

    def _generate_single_chunk(
        self, 
        text, 
        cfg_weight, 
        max_new_tokens, 
        temperature, 
        max_cache_len, 
        repetition_penalty, 
        min_p, 
        top_p, 
        disable_tqdm,
        t3_params
    ):
        """Generate audio for a single text chunk."""
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        # Use shared text token preparation
        text_tokens = prepare_text_tokens(
            text_tokens, 
            self.t3.hp.start_text_token, 
            self.t3.hp.stop_text_token, 
            cfg_weight
        )

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                cfg_weight=cfg_weight,
                max_cache_len=max_cache_len,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                disable_tqdm=disable_tqdm,
                **t3_params,
            )

            def speech_to_wav(speech_tokens):
                # Extract only the conditional batch.
                speech_tokens = speech_tokens[0]

                speech_tokens = drop_invalid_tokens(speech_tokens)
                speech_tokens = drop_bad_tokens(speech_tokens)
                
                wav, _ = self.s3gen.inference(
                    speech_tokens=speech_tokens,
                    ref_dict=self.conds.gen,
                )

                return wav
            return speech_to_wav(speech_tokens)

