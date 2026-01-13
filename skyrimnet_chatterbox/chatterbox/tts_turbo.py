from .shared_audio_utils import load_and_preprocess_audio
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
    concatenate_audio_tensors,
    get_map_location
)
from .tensor_utils import (
    load_conditionals_safe,
    load_voice_encoder_safe,
)
import os
import math
from pathlib import Path

import torch

from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from .conditionals import Conditionals
from .models.t3 import T3
from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond
from .models.t3.modules.t3_config import T3Config
from .models.s3gen.const import S3GEN_SIL
import logging
logger = logging.getLogger(__name__)


REPO_ID = "ResembleAI/chatterbox-turbo"


class ChatterboxTurboTTS:
    ENC_COND_LEN = 15 * S3_SR
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
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTurboTTS':
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        map_location = get_map_location(device)

        ve = load_voice_encoder_safe(ckpt_dir, device, is_multilingual=False)

        # Turbo specific hp
        hp = T3Config(text_tokens_dict_size=50276)
        hp.llama_config_name = "GPT2_medium"
        hp.speech_tokens_dict_size = 6563
        hp.input_pos_emb = None
        hp.speech_cond_prompt_len = 375
        hp.use_perceiver_resampler = False
        hp.emotion_adv = False

        t3 = T3(hp)
        t3_state = load_file(ckpt_dir / "t3_turbo_v1.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        del t3.tfmr.wte
        t3.to(device).eval()

        s3gen = S3Gen(meanflow=True)
        weights = load_file(ckpt_dir / "s3gen_meanflow.safetensors")
        s3gen.load_state_dict(
            weights, strict=True
        )
        s3gen.to(device).eval()
        
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if len(tokenizer) != 50276:
            print(f"WARNING: Tokenizer len {len(tokenizer)} != 50276")

        # Load conditionals if they exist
        conds = load_conditionals_safe(ckpt_dir, device, is_multilingual=False)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTurboTTS':
        # Check if MPS is available on macOS
        device = check_mps_availability(device)

        local_path = snapshot_download(
            repo_id=REPO_ID,
            token=os.getenv("HF_TOKEN"),
            # Optional: Filter to download only what you need
            allow_patterns=["*.safetensors",
                            "*.json", "*.txt", "*.pt", "*.model"],
            cache_dir="models"
        )

        return cls.from_local(local_path, device)

    def set_conditionals(self, conds: Conditionals):
        """
        Set the conditionals for T3 and S3Gen.
        """
        if conds is not None:
            self.conds = conds.to(self.device)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5, norm_loudness=True):
        # Use shared audio preprocessing
        s3gen_ref_wav, ref_16k_wav_tensor = load_and_preprocess_audio(
            wav_fpath=wav_fpath, device=self.device, min_duration=5.0)

        # Slice as tensor
        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(
            s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward(
                [ref_16k_wav_tensor[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(
                t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        # ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = self.ve.embeds_from_wavs_tensor(
            [ref_16k_wav_tensor], sample_rate=S3_SR)
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration *
            torch.ones(1, 1, 1, dtype=ve_embed.dtype),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        repetition_penalty=1.2,
        min_p=0.0,
        top_p=0.95,
        top_k=1000,
        audio_prompt_path=None,
        exaggeration=0.0,
        cfg_weight=0.0,
        temperature=0.8,
        norm_loudness=True,
        # Optimization parameters
        max_gen_len=1000,        
        disable_tqdm=False,
        t3_params={},
    ):
        if audio_prompt_path:
            self.prepare_conditionals(
                audio_prompt_path, exaggeration=exaggeration, norm_loudness=norm_loudness)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Norm and tokenize text
        text = punc_norm(text)
        #print(f"Normalized text: {text}")
        # Check if text needs to be chunked based on token count
        estimated_tokens = estimate_token_count(text, self.tokenizer)

        max_chunk_tokens = 80 #max(200, max_cache_len - max_gen_len - 75)

        if estimated_tokens <= max_chunk_tokens:
            # Text is small enough - process normally without chunking
            return self._generate_single_chunk(
                text=text, temperature=temperature, repetition_penalty=repetition_penalty, top_p=top_p, top_k=top_k, max_gen_len=max_gen_len, disable_tqdm=disable_tqdm, t3_params=t3_params
            )
        else:
            # Text is too large - split into chunks and process separately
            text_chunks = smart_text_splitter(text, max_chunk_tokens, self.tokenizer)

            # Pre-tokenize all chunks upfront (CPU-bound, can be parallelized)
            tokenized_chunks = [
                self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)
                for chunk in text_chunks
            ]

            # Process chunks with pipelining for better GPU utilization
            # Note: Threading with CUDA graphs has TLS issues, so we use sequential processing
            # which allows torch.compile optimizations to work properly
            audio_chunks = self._generate_chunks_sequential(
                tokenized_chunks=tokenized_chunks, temperature=temperature, 
                repetition_penalty=repetition_penalty, top_p=top_p, top_k=top_k, 
                max_gen_len=max_gen_len, disable_tqdm=disable_tqdm, t3_params=t3_params
            )
            
            # Concatenate all audio chunks with brief silence between them
            return concatenate_audio_tensors(audio_chunks, silence_duration=0.12, sample_rate=self.sr)

    def _generate_single_chunk(
        self,
        text,
        temperature,
        repetition_penalty,
        top_p,
        top_k,
        max_gen_len,
        disable_tqdm,
        t3_params
    ):
        text_tokens = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True)
        text_tokens = text_tokens.input_ids.to(self.device)

        return self._generate_from_tokens(
            text_tokens=text_tokens, temperature=temperature, repetition_penalty=repetition_penalty, top_p=top_p, top_k=top_k, max_gen_len=max_gen_len, disable_tqdm=disable_tqdm, t3_params=t3_params
        )

    def _generate_from_tokens(
        self,
        text_tokens,
        temperature,
        repetition_penalty,
        top_p,
        top_k,
        max_gen_len,
        disable_tqdm,
        t3_params
    ):
        """Generate audio from pre-tokenized text (avoids redundant tokenization)."""
        # Default to "eager" backend to avoid torch.compile issues with CUDA graphs
        # The "reduce-overhead" backend can cause threading/TLS issues and dynamo errors
        #inference_params = {'generate_token_backend': 'eager', **t3_params}
        
        speech_tokens = self.t3.inference_turbo(
            t3_cond=self.conds.t3,
            text_tokens=text_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_gen_len=max_gen_len,
            disable_tqdm=disable_tqdm,
            #**inference_params,
            **t3_params
        )

        # Remove OOV tokens and add silence to end
        speech_tokens = speech_tokens[speech_tokens < 6561]
        speech_tokens = speech_tokens.to(self.device)
        silence = torch.tensor(
            [S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]).long().to(self.device)
        speech_tokens = torch.cat([speech_tokens, silence])

        wav, _ = self.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=self.conds.gen,
            n_cfm_timesteps=2,
        )
        return wav

    def _generate_chunks_sequential(self, tokenized_chunks, temperature, repetition_penalty,
                                     top_p, top_k, max_gen_len, disable_tqdm, t3_params):
        """
        Process text chunks sequentially for maximum throughput.
        
        This is the fastest approach for batch processing where you need
        the complete audio as quickly as possible (not real-time streaming).
        """
        audio_chunks = []
        silence = torch.tensor([S3GEN_SIL] * 3, dtype=torch.long, device=self.device)
        
        for text_tokens in tokenized_chunks:
            # T3 inference - generate all speech tokens for this text chunk
            speech_tokens = self.t3.inference_turbo(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_gen_len=max_gen_len,
                disable_tqdm=disable_tqdm,
                **t3_params,
            )
            
            # Clean up tokens and add silence
            speech_tokens = speech_tokens[speech_tokens < 6561]
            speech_tokens = torch.cat([speech_tokens.to(self.device), silence])
            
            # S3Gen inference - single call per chunk (most efficient)
            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
                n_cfm_timesteps=2,
            )
            audio_chunks.append(wav)
        
        return audio_chunks

    def _generate_chunks_pipelined(self, tokenized_chunks, temperature, repetition_penalty, 
                                    top_p, top_k, max_gen_len, disable_tqdm, t3_params):
        """
        Pipeline T3 and S3Gen using threads + CUDA streams for true parallelism.
        
        Architecture:
        - Producer thread: Runs T3 inference, puts speech_tokens in queue
        - Consumer thread (main): Runs S3Gen inference on completed tokens
        
        This achieves overlap because Python threads release the GIL during CUDA operations.
        """
        import queue
        import threading

        if not torch.cuda.is_available() or len(tokenized_chunks) < 2:
            # Fall back to sequential for CPU or single chunk
            return [self._generate_from_tokens(
                text_tokens=t, temperature=temperature, repetition_penalty=repetition_penalty, 
                top_p=top_p, top_k=top_k, max_gen_len=max_gen_len, 
                disable_tqdm=disable_tqdm, t3_params=t3_params) 
                for t in tokenized_chunks]

        # Create separate CUDA streams
        t3_stream = torch.cuda.Stream()
        s3gen_stream = torch.cuda.Stream()
        
        # Queue for passing speech tokens from T3 to S3Gen
        # maxsize=2 allows T3 to work 1 chunk ahead
        token_queue = queue.Queue(maxsize=2)
        
        silence = torch.tensor([S3GEN_SIL] * 3, dtype=torch.long, device=self.device)
        t3_cond = self.conds.t3
        
        # Override t3_params for pipelined mode:
        # - Use "eager" backend because torch.compile's CUDA graphs use thread-local storage
        #   that isn't initialized in the producer thread
        # - Disable benchmark/tqdm to reduce overhead
        pipelined_t3_params = {
            **t3_params, 
            'benchmark': False,
            'generate_token_backend': 'eager',  # CUDA graphs don't work in separate threads
        }
        
        def t3_producer():
            """Run T3 inference for all chunks and queue the results."""
            try:
                with torch.cuda.stream(t3_stream):
                    for i, text_tokens in enumerate(tokenized_chunks):
                        speech_tokens = self.t3.inference_turbo(
                            t3_cond=t3_cond,
                            text_tokens=text_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            repetition_penalty=repetition_penalty,
                            max_gen_len=max_gen_len,
                            disable_tqdm=True,  # Always disable tqdm in producer thread
                            **pipelined_t3_params,
                        )
                        speech_tokens = speech_tokens[speech_tokens < 6561]
                        speech_tokens = torch.cat([speech_tokens, silence])
                        
                        # Record event so S3Gen knows when this chunk's tokens are ready
                        ready_event = torch.cuda.Event()
                        ready_event.record(t3_stream)
                        
                        # Put in queue (blocks if queue is full - provides backpressure)
                        token_queue.put((speech_tokens, ready_event))
            except Exception as e:
                logger.error(f"T3 producer thread error: {e}")
                token_queue.put(('error', e))
            finally:
                # Signal completion
                token_queue.put(None)
        
        # Start T3 producer thread
        producer_thread = threading.Thread(target=t3_producer, daemon=True)
        producer_thread.start()
        
        # S3Gen consumer runs on main thread with its own stream
        audio_chunks = []
        ref_dict = self.conds.gen
        
        with torch.cuda.stream(s3gen_stream):
            while True:
                item = token_queue.get()
                if item is None:
                    break
                
                # Check for error from producer
                if isinstance(item, tuple) and item[0] == 'error':
                    raise item[1]
                    
                speech_tokens, ready_event = item
                
                # Wait for T3 to finish this chunk's tokens before S3Gen uses them
                ready_event.wait(s3gen_stream)
                
                wav, _ = self.s3gen.inference(
                    speech_tokens=speech_tokens,
                    ref_dict=ref_dict,
                    n_cfm_timesteps=2,
                )
                audio_chunks.append(wav)
        
        # Wait for producer thread to finish
        producer_thread.join()
        
        # Ensure all GPU work is complete
        torch.cuda.synchronize()

        return audio_chunks

    def generate_streaming(
        self,
        text,
        repetition_penalty=1.2,
        min_p=0.0,
        top_p=0.95,
        top_k=1000,
        audio_prompt_path=None,
        exaggeration=0.0,
        cfg_weight=0.0,
        temperature=0.8,
        norm_loudness=True,
        max_gen_len=1000,
        disable_tqdm=True,
        t3_params={},
        # Streaming parameters
        token_chunk_size=100,  # Yield audio after this many speech tokens
    ):
        """
        Streaming TTS generation - yields audio chunks as they're generated.
        
        This provides lower time-to-first-audio by converting speech tokens to audio
        incrementally rather than waiting for all tokens to be generated.
        
        Automatically splits long texts into chunks to handle the model's context limit,
        then streams audio from each chunk.
        
        Args:
            token_chunk_size: Number of speech tokens to accumulate before converting to audio.
                             Lower = faster first audio, Higher = better quality/efficiency.
                             Recommended: 80-150 tokens (~1-2 seconds of audio)
        
        Yields:
            torch.Tensor: Audio waveform chunks (can be played/saved incrementally)
        
        Example:
            audio_chunks = []
            for chunk in model.generate_streaming(text, audio_prompt_path=path):
                audio_chunks.append(chunk)
                # Optional: play chunk immediately for real-time output
            full_audio = concatenate_audio_tensors(audio_chunks, silence_duration=0.05, sample_rate=24000)
        """
        if audio_prompt_path:
            self.prepare_conditionals(
                audio_prompt_path, exaggeration=exaggeration, norm_loudness=norm_loudness)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Norm text
        text = punc_norm(text)
        
        # Check if text needs splitting
        estimated_tokens = estimate_token_count(text, self.tokenizer)
        max_chunk_tokens = 80  # Same as generate()
        
        if estimated_tokens <= max_chunk_tokens:
            # Single chunk - stream directly
            text_chunks = [text]
        else:
            # Split into multiple chunks for model context limit
            text_chunks = smart_text_splitter(text, max_chunk_tokens, self.tokenizer)
        print(f"Text split into {len(text_chunks)} chunks")
        silence = torch.tensor([S3GEN_SIL] * 3, dtype=torch.long, device=self.device)
        
        for chunk_idx, text_chunk in enumerate(text_chunks):
            text_tokens = self.tokenizer(
                text_chunk, return_tensors="pt", padding=True, truncation=True
            ).input_ids.to(self.device)

            # Use the streaming T3 inference for this chunk
            # Filter out params not supported by streaming inference
            streaming_t3_params = {k: v for k, v in t3_params.items() 
                                   if k in ('generate_token_backend', 'skip_when_1')}
            
            for speech_token_chunk in self.t3.inference_turbo_streaming(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                max_gen_len=max_gen_len,
                disable_tqdm=disable_tqdm,
                token_chunk_size=token_chunk_size,
                **streaming_t3_params,
            ):
                # Clean up tokens
                speech_tokens = speech_token_chunk[speech_token_chunk < 6561]
                if len(speech_tokens) == 0:
                    continue
                    
                speech_tokens = torch.cat([speech_tokens.to(self.device), silence])
                
                # Convert to audio immediately
                wav, _ = self.s3gen.inference(
                    speech_tokens=speech_tokens,
                    ref_dict=self.conds.gen,
                    n_cfm_timesteps=2,
                )
                yield wav

    def generate_streaming_collected(
        self,
        text,
        silence_duration=0.05,
        **kwargs
    ):
        """
        Streaming generation that collects and concatenates all chunks.
        
        This provides the same output as generate() but with streaming benefits:
        - Lower memory peak (processes incrementally)
        - Can be used for progress tracking
        
        Args:
            text: Text to synthesize
            silence_duration: Silence between chunks in seconds
            **kwargs: Same arguments as generate_streaming()
            
        Returns:
            torch.Tensor: Complete audio waveform
        """
        audio_chunks = list(self.generate_streaming(text, **kwargs))
        if not audio_chunks:
            return torch.tensor([], device=self.device)
        return concatenate_audio_tensors(audio_chunks, silence_duration=silence_duration, sample_rate=self.sr)