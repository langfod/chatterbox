# Copyright (c) 2025 Resemble AI
# MIT License
import logging
from typing import Optional, Tuple, Callable
import time

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from transformers.cache_utils import StaticCache

from transformers import LlamaModel, LlamaConfig, GPT2Config, GPT2Model
from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor,
    TopPLogitsWarper,
    MinPLogitsWarper,
)
from .modules.learned_pos_emb import LearnedPositionEmbeddings

from .modules.cond_enc import T3CondEnc, T3Cond
from .modules.t3_config import T3Config
from .llama_configs import LLAMA_CONFIGS
from .inference.t3_hf_backend import T3HuggingfaceBackend

from .fast_min_p_warper import FastMinPLogitsWarper
from .fast_top_p_warper import FastTopPLogitsWarper
from .fast_top_k_warper import FastTopKLogitsWarper
from .t3_cuda_graphs import T3StepCUDAGraphWrapper, get_next_bucket

logger = logging.getLogger(__name__)

TOKEN_LIMIT = 1500


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def _ensure_BOT_EOT(text_tokens: Tensor, hp):
    B = text_tokens.size(0)
    assert (text_tokens == hp.start_text_token).int(
    ).sum() >= B, "missing start_text_token"
    assert (text_tokens == hp.stop_text_token).int(
    ).sum() >= B, "missing stop_text_token"


class T3(nn.Module):
    """
    Token-To-Token (T3) TTS model using huggingface transformer models as backbones,
        * tokenization, including start / stop tokens are always added externally to this class
        * conditioning data like CLAP, emotion, etc are all in a separate file for more modularity
        * careful! this class assumes relative positional encoding -- with absolute PE, we would at
            least want to reset the position to 0 when speech tokens begin, and optionally use a
            different PE embedding space for speech.
    """

    def __init__(self, hp=None):
        if hp is None:
            hp = T3Config.english_only()
        super().__init__()
        self.hp = hp

        config_dict = LLAMA_CONFIGS[hp.llama_config_name]
        self.is_gpt = config_dict.get("model_type") == "gpt2"

        if self.is_gpt:
            self.cfg = GPT2Config(**config_dict)
            self.tfmr = GPT2Model(self.cfg)
        else:
            self.cfg = LlamaConfig(**config_dict)
            self.tfmr = LlamaModel(self.cfg)

        self.dim = self.cfg.hidden_size
        self.deepspeed_patch_applied = False

        # conditioning / embedding
        self.cond_enc = T3CondEnc(hp)
        self.text_emb = nn.Embedding(hp.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(hp.speech_tokens_dict_size, self.dim)

        # custom position embedding
        self.text_pos_emb = None
        self.speech_pos_emb = None
        if hp.input_pos_emb == "learned":
            max_text_seq_len = hp.max_text_tokens + 2
            self.text_pos_emb = LearnedPositionEmbeddings(
                max_text_seq_len, self.dim)

            max_mel_seq_len = hp.max_speech_tokens + 2 + 2
            self.speech_pos_emb = LearnedPositionEmbeddings(
                max_mel_seq_len, self.dim)

        # logit projection
        self.text_head = nn.Linear(
            self.cfg.hidden_size, hp.text_tokens_dict_size, bias=False)
        self.speech_head = nn.Linear(
            self.cfg.hidden_size, hp.speech_tokens_dict_size, bias=self.is_gpt)
        self.compiled = False
        self.init_processors()

    @property
    def device(self):
        return self.speech_emb.weight.device

    @property
    def dtype(self):
        return self.speech_emb.weight.dtype

    def to(self, *args, **kwargs):
        # Move warper tensors to the target device
        if hasattr(self.min_p_warper, 'false_tensor'):
            self.min_p_warper.false_tensor = self.min_p_warper.false_tensor.to(
                *args, **kwargs)
        if hasattr(self.min_p_warper, 'min_p_tensor'):
            self.min_p_warper.min_p_tensor = self.min_p_warper.min_p_tensor.to(
                *args, **kwargs)
        if hasattr(self.top_p_warper, 'zero_tensor'):
            self.top_p_warper.zero_tensor = self.top_p_warper.zero_tensor.to(
                *args, **kwargs)
        return super().to(*args, **kwargs)

    def prepare_conditioning(self, t3_cond: T3Cond):
        """
        Token cond data needs to be embedded, so that needs to be here instead of in `T3CondEnc`.
        """
        if t3_cond.cond_prompt_speech_tokens is not None and t3_cond.cond_prompt_speech_emb is None:
            cond_emb = self.speech_emb(t3_cond.cond_prompt_speech_tokens)
            if self.speech_pos_emb is not None:
                cond_emb = cond_emb + \
                    self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
            t3_cond.cond_prompt_speech_emb = cond_emb
        return self.cond_enc(t3_cond)  # (B, len_cond, dim)

    def prepare_input_embeds(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        cfg_weight: float = 0.0,
    ):
        if self.dtype != t3_cond.speaker_emb.dtype:
            t3_cond.to(dtype=self.dtype)
        # prepare input embeddings (skip backbone tranformer embeddings)
        cond_emb = self.prepare_conditioning(t3_cond)  # (B, len_cond, dim)
        text_emb = self.text_emb(text_tokens)  # (B, len_text, dim)
        if cfg_weight > 0.0:
            text_emb[1].zero_()  # CFG uncond

        speech_emb = self.speech_emb(speech_tokens)  # (B, len_speech, dim)
        if self.hp.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens)
            speech_emb = speech_emb + self.speech_pos_emb(speech_tokens)
        len_cond = cond_emb.size(1)

        if cond_emb.size(0) != text_emb.size(0):
            cond_emb = cond_emb.expand(text_emb.size(0), -1, -1)

        # concat
        embeds = torch.stack([
            torch.cat((ce, te, se))
            for ce, te, se in zip(cond_emb, text_emb, speech_emb)
        ])  # (B, length, dim)
        return embeds, len_cond

    def forward(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        speech_token_lens: torch.LongTensor,
        training=False,
    ):
        _ensure_BOT_EOT(text_tokens, self.hp)

        # prepare custom input embeds
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_tokens,
        )

        # backbone tranformer forward
        tfmr_out = self.tfmr.forward(
            input_ids=None,
            # position_ids=position_ids, # TODO? ROPE should be fine?
            inputs_embeds=embeds,
            output_hidden_states=True,
            return_dict=True,
            use_cache=(not training),
        )
        # final tfmr layer output, (B, seq, dim)
        hidden_states = tfmr_out.hidden_states[-1]

        # post-processing: splice out text and speech parts of hidden states
        len_text = text_tokens.size(1)
        len_speech = speech_tokens.size(1)
        B, _, dim = hidden_states.shape
        device, dtype = hidden_states.device, hidden_states.dtype
        text_latents = torch.zeros(
            B, len_text, dim, dtype=dtype, device=device)
        speech_latents = torch.zeros(
            B, len_speech, dim, dtype=dtype, device=device)
        ttl, stl = text_token_lens, speech_token_lens
        for i in range(B):
            text_end = len_cond + ttl[i].item()
            speech_start = len_cond + text_tokens.size(1)
            speech_end = speech_start + stl[i].item()
            text_latents[i, :ttl[i]] = hidden_states[i, len_cond:text_end]
            speech_latents[i, :stl[i]] = hidden_states[i,
                                                       speech_start:speech_end]

        # logit projection
        text_logits = self.text_head(text_latents)
        speech_logits = self.speech_head(speech_latents)

        return AttrDict(
            text_logits=text_logits,
            text_latents=text_latents,
            speech_logits=speech_logits,
            speech_latents=speech_latents,
            hidden_states=hidden_states,
        )

    def loss(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        speech_token_lens: torch.LongTensor,
    ):
        "training method"
        len_text = text_tokens.size(1)
        len_speech = speech_tokens.size(1)
        assert len_text == text_token_lens.max()
        assert len_speech == speech_token_lens.max()

        out = self.forward(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            training=True,
        )  # (B, seq, vocab_size)

        # Calc CCE losses
        IGNORE_ID = -100
        device = out.text_logits.device
        mask_text = torch.arange(len_text, device=device)[
            None] >= text_token_lens[:, None]  # (B, len_text)
        mask_speech = torch.arange(len_speech, device=device)[
            None] >= speech_token_lens[:, None]  # (B, len_speech)
        masked_text = text_tokens.masked_fill(mask_text, IGNORE_ID)
        masked_speech = speech_tokens.masked_fill(mask_speech, IGNORE_ID)
        loss_text = F.cross_entropy(
            out.text_logits, masked_text, ignore_index=IGNORE_ID)
        loss_speech = F.cross_entropy(
            out.speech_logits, masked_speech, ignore_index=IGNORE_ID)

        return loss_text, loss_speech

    def init_patched_model(self):
        # TODO? synchronize the expensive compile function
        # with self.compile_lock:
        if not self.compiled:
            # alignment_stream_analyzer = AlignmentStreamAnalyzer(
            #     self.tfmr,
            #     None,
            #     text_tokens_slice=(len_cond, len_cond + text_tokens.size(-1)),
            #     alignment_layer_idx=9, # TODO: hparam or something?
            #     eos_idx=self.hp.stop_speech_token,
            # )
            patched_model = T3HuggingfaceBackend(
                config=self.cfg,
                llama=self.tfmr,
                speech_enc=self.speech_emb,
                speech_head=self.speech_head,
                # alignment_stream_analyzer=alignment_stream_analyzer,
            )
            self.patched_model = patched_model
            self.compiled = True

    def get_cache(self, config, max_batch_size, max_cache_len, device, dtype):
        if hasattr(self, 'backend_cache'):
            if self.backend_cache_params['max_batch_size'] == max_batch_size and \
                    self.backend_cache_params['max_cache_len'] == max_cache_len and \
                    self.backend_cache_params['dtype'] == dtype and \
                    self.backend_cache_params['device'] == device:
                self.backend_cache.reset()
                return self.backend_cache
            else:
                del self.backend_cache

        cache = StaticCache(
            config=config,
            max_batch_size=max_batch_size,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
        )
        # save parameters in t3 since huggingface fails deprecation standards.
        self.backend_cache_params = {
            'max_batch_size': max_batch_size,
            'max_cache_len': max_cache_len,
            'device': device,
            'dtype': dtype,
        }
        self.backend_cache = cache
        return cache

    def get_speech_pos_embedding_cache(self, max_gen_tokens, dtype):
        if not hasattr(self, '_speech_pos_embedding_cache') or self._speech_pos_embedding_cache.size(0) < max_gen_tokens:
            # Create cache with embeddings for positions 0 to max_gen_tokens-1
            self._speech_pos_embedding_cache = []
            for pos in range(max_gen_tokens):
                embedding = self.speech_pos_emb.get_fixed_embedding(pos)
                self._speech_pos_embedding_cache.append(embedding)
            # Stack and move to device
            self._speech_pos_embedding_cache = torch.stack(
                self._speech_pos_embedding_cache, dim=0).to(device=self.device)
        elif self._speech_pos_embedding_cache.dtype != dtype:
            self._speech_pos_embedding_cache = self._speech_pos_embedding_cache.to(
                dtype=dtype)
        return self._speech_pos_embedding_cache

    def init_speech_embedding_cache(self, vocab_size, dtype):
        if not hasattr(self, '_speech_embedding_cache') or self._speech_embedding_cache.size(0) < vocab_size:
            # Create cache with embeddings for positions 0 to max_gen_tokens-1
            self._speech_embedding_cache = []
            for pos in range(vocab_size):
                pos = torch.tensor([pos], device=self.device)
                embedding = self.speech_emb(pos)
                self._speech_embedding_cache.append(embedding.squeeze(0))
            # Stack and move to device
            self._speech_embedding_cache = torch.stack(
                self._speech_embedding_cache, dim=0).to(device=self.device)
        elif self._speech_embedding_cache.dtype != dtype:
            self._speech_embedding_cache = self._speech_embedding_cache.to(
                dtype=dtype)
        return self._speech_embedding_cache

    def init_processors(self, top_k=1000, top_p=1.0, min_p=0.05, repetition_penalty=1.2):
        # Processors should be pre-instantiated to avoid recompilation
        self.top_k_warper = FastTopKLogitsWarper(top_k=top_k)
        self.top_p_warper = FastTopPLogitsWarper(top_p=top_p)
        self.min_p_warper = FastMinPLogitsWarper(min_p=min_p)
        self.repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(
            penalty=repetition_penalty)

    def update_processors(self, top_p, min_p, repetition_penalty, skip_when_1=False, top_k=None):
        # Ensure warper tensors are on the correct device
        device = self.device
        if hasattr(self.min_p_warper, 'false_tensor') and self.min_p_warper.false_tensor.device != device:
            self.min_p_warper.false_tensor = self.min_p_warper.false_tensor.to(
                device)
        if hasattr(self.min_p_warper, 'min_p_tensor') and self.min_p_warper.min_p_tensor.device != device:
            self.min_p_warper.min_p_tensor = self.min_p_warper.min_p_tensor.to(
                device)
        if hasattr(self.top_p_warper, 'zero_tensor') and self.top_p_warper.zero_tensor.device != device:
            self.top_p_warper.zero_tensor = self.top_p_warper.zero_tensor.to(
                device)

        if top_k is not None and self.top_k_warper.top_k != top_k:
            self.top_k_warper.top_k = top_k
        if self.top_p_warper.top_p != top_p:
            self.top_p_warper.top_p = top_p
            self.top_p_warper.skip_when_1 = skip_when_1
        if self.min_p_warper.min_p != min_p:
            self.min_p_warper.min_p = min_p
            # Update the pre-allocated tensor as well
            if hasattr(self.min_p_warper, 'min_p_tensor'):
                self.min_p_warper.min_p_tensor = torch.tensor(
                    min_p, device=device)
        if self.repetition_penalty_processor.penalty != repetition_penalty:
            self.repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(
                penalty=repetition_penalty)

    @torch.inference_mode()
    def inference(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: Tensor,
        initial_speech_tokens: Optional[Tensor] = None,

        # misc conditioning
        prepend_prompt_speech_tokens: Optional[Tensor] = None,

        # HF generate args
        num_return_sequences=1,
        max_new_tokens=None,
        stop_on_eos=True,
        do_sample=True,
        temperature=0.8,
        min_p=0.05,
        top_p=1.0,
        length_penalty=1.0,
        repetition_penalty=1.2,
        cfg_weight=0,
        disable_tqdm=False,
        # optimizations
        max_cache_len=None,
        initial_forward_pass_backend="eager",
        generate_token_backend="cudagraphs-manual",
        stride_length=4,
        skip_when_1=True,
        benchmark=False,
    ):
        """
        Args:
            text_tokens: a 1D (unbatched) or 2D (batched) tensor.
        """
        # Validate / sanitize inputs
        assert prepend_prompt_speech_tokens is None, "not implemented"
        _ensure_BOT_EOT(text_tokens, self.hp)
        text_tokens = torch.atleast_2d(text_tokens).to(
            dtype=torch.long, device=self.device)

        # Default initial speech to a single start-of-speech token
        if initial_speech_tokens is None:
            initial_speech_tokens = self.hp.start_speech_token * \
                torch.ones_like(text_tokens[:, :1])

        # Prepare custom input embeds
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=initial_speech_tokens,
            cfg_weight=cfg_weight,
        )

        # In order to use the standard HF generate method, we need to extend some methods to inject our custom logic
        # Note the llama-specific logic. Other tfmr types can be added later.
        self.init_patched_model()
        # Pre-compute embeddings cache for the generation loop
        self.get_speech_pos_embedding_cache(
            TOKEN_LIMIT + 1 or self.hp.max_speech_tokens, dtype=embeds.dtype)
        self.init_speech_embedding_cache(
            vocab_size=self.hp.speech_tokens_dict_size, dtype=embeds.dtype)

        # # Run normal generate method, which calls our custom extended methods
        # return self.patched_model.generate(
        #     inputs=initial_speech_tokens,
        #     decoder_cond=embeds,
        #     bos_token_id=self.hp.start_speech_token,
        #     eos_token_id=(self.hp.stop_speech_token if stop_on_eos else -1),
        #     pad_token_id=self.hp.stop_speech_token,
        #     max_new_tokens=max_new_tokens or self.hp.max_speech_tokens,
        #     num_return_sequences=num_return_sequences,
        #     temperature=temperature,
        #     top_p=top_p,
        #     length_penalty=length_penalty,
        #     repetition_penalty=repetition_penalty,
        #     do_sample=do_sample,
        #     # cache_implementation=None if not self.compiled else "static",
        # )

        device = embeds.device

        bos_token = torch.tensor(
            [[self.hp.start_speech_token]], dtype=torch.long, device=device)
        bos_embed = self._speech_embedding_cache[bos_token]
        bos_embed = bos_embed + self._speech_pos_embedding_cache[0]

        # batch_size=2 for CFG
        bos_embed = torch.cat([bos_embed, bos_embed])

        # Combine condition and BOS token for the initial input if cfg_weight > 0
        if cfg_weight > 0:
            inputs_embeds = torch.cat([embeds, bos_embed], dim=1)
        else:
            inputs_embeds = embeds

        # Track generated token ids; start with the BOS token.
        PAD_TOKEN_ID = self.hp.stop_speech_token + 1  # Assuming unused
        bos_len = bos_token.shape[1]  # == 1

        # Instantiate the logits processors.
        self.update_processors(
            top_p, min_p, repetition_penalty, skip_when_1=skip_when_1)

        # move all inputs to patched_model.dtype
        inputs_embeds = inputs_embeds.to(self.patched_model.dtype)
        embeds = embeds.to(self.patched_model.dtype)
        bos_embed = bos_embed.to(self.patched_model.dtype)

        stop_token_tensor = torch.tensor(
            self.hp.stop_speech_token, device=self.device)

        # Fix: Set max_batch_size based on CFG usage
        effective_batch_size = 2 if cfg_weight > 0.0 else 1

        _, seq_len = inputs_embeds.shape[:2]
        if max_cache_len < seq_len + max_new_tokens:
            print(
                f"Warning: max_cache_len {max_cache_len} is too small for seq_len {seq_len} and max_new_tokens {max_new_tokens}")
            print(f"Reducing max_new_tokens to {max_cache_len - seq_len}")
            max_new_tokens = max_cache_len - seq_len

        # using batch size of 1, otherwise use generated_ids[:, i]
        assert max_new_tokens < TOKEN_LIMIT, f"max_new_tokens {max_new_tokens} is too large, maximum is {TOKEN_LIMIT}"
        generated_ids = torch.full(
            (1, bos_len + TOKEN_LIMIT), PAD_TOKEN_ID, dtype=torch.long, device=device)
        generated_ids[0, :bos_len] = bos_token

        kv_cache = self.get_cache(
            config=self.patched_model.config,
            max_batch_size=effective_batch_size,
            max_cache_len=max_cache_len,
            device=self.patched_model.device,
            dtype=self.patched_model.dtype,
        )

        # Move check higher to avoid polluting the loop
        assert not kv_cache.get_seq_length() > 0, \
            "Cannot process large input when cache already has content"

        length_guesstimate = text_tokens.shape[1] * 2
        # print(f"Estimated token count: {length_guesstimate}")

        # ---- Pad input_embeds to fixed length for compilation stability ----
        # This ensures that input_embeds always has the same shape for torch.compile
        def pad_to_fixed_length(inputs_embeds: Tensor, TOKEN_LIMIT: int):
            PADDED_SEQ_LEN = TOKEN_LIMIT  # max possible length
            pad_len = PADDED_SEQ_LEN - inputs_embeds.shape[1]
            pad_shape = list(inputs_embeds.shape)
            pad_shape[1] = pad_len
            pad_embeds = torch.zeros(
                pad_shape, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            inputs_embeds = torch.cat([inputs_embeds, pad_embeds], dim=1)

            return inputs_embeds

        # print(f"Input embeds shape before padding: {inputs_embeds.shape}")

        inputs_embeds = pad_to_fixed_length(inputs_embeds, TOKEN_LIMIT)

        # ---- Initial Forward Pass (no kv_cache yet) ----
        initial_forward_pass = _initial_forward_pass_variants.get(
            initial_forward_pass_backend, _initial_forward_pass_variants["eager"])

        if benchmark:
            import sys
            _t3_init_start = time.time()
            print(f"[T3_TIMING] Starting initial forward pass with backend: {initial_forward_pass_backend}", flush=True)
            sys.stdout.flush()

        output_logits = initial_forward_pass(
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
            seq_len=seq_len,
            patched_model=self.patched_model
        ).clone()  # Clone to avoid in-place modification issues

        if benchmark:
            _t3_init_end = time.time()
            print(f"[T3_TIMING] Initial forward pass completed in {_t3_init_end - _t3_init_start:.3f}s", flush=True)

        indices = torch.arange(1, max_new_tokens + 1,
                               device=generated_ids.device)
        batch_idx = torch.zeros(1, dtype=torch.long,
                                device=generated_ids.device)
        if generate_token_backend == "cudagraphs-manual":
            # Track CFG mode and recreate wrapper if it changes (different batch dimensions)
            current_cfg_mode = 1 if cfg_weight > 0.0 else 0
            if not hasattr(self, "cudagraph_wrapper") or \
               not hasattr(self, "_cudagraph_cfg_mode") or \
               self._cudagraph_cfg_mode != current_cfg_mode:
                if hasattr(self, "cudagraph_wrapper"):
                    logger.debug(
                        f"Recreating CUDA graph wrapper due to CFG mode change: {getattr(self, '_cudagraph_cfg_mode', 'None')} -> {current_cfg_mode}")
                if benchmark:
                    print(f"[T3_TIMING] Creating new CUDA graph wrapper (cfg_mode: {current_cfg_mode})", flush=True)
                self.cudagraph_wrapper = T3StepCUDAGraphWrapper(
                    generate_t3_token,
                    self.patched_model,
                    kv_cache,
                    self.repetition_penalty_processor,
                    self.min_p_warper,
                    self.top_p_warper,
                )
                self._cudagraph_cfg_mode = current_cfg_mode
            self.cudagraph_wrapper.guard(cfg_weight)
            _generate_token_variants["cudagraphs-manual"] = self.cudagraph_wrapper
        generate_token = _generate_token_variants.get(
            generate_token_backend, _generate_token_variants["eager"])

        if benchmark:
            print(
            f"[T3_TIMING] Starting token generation loop with backend: {generate_token_backend}, max_tokens: {max_new_tokens}", flush=True)
            _t3_loop_start = time.time()        
            start = time.time()
            torch.cuda.synchronize()  # For benchmarking to have correct it/s
        stride_length = stride_length if "stride" in generate_token_backend else 1
        for i in tqdm(range(max_new_tokens // stride_length), desc="Sampling", dynamic_ncols=True, disable=disable_tqdm):
            i_tensor = indices[i * stride_length]
            # Check for EOS token.
            if i * stride_length > length_guesstimate and i % (20 // stride_length) == 0:
                if (generated_ids == stop_token_tensor).any():
                    if benchmark:
                        torch.cuda.synchronize()  # For benchmarking to have correct it/s
                        print(
                            f"Stopping at {(i + 1) * stride_length} because EOS token was generated")
                        print(
                            f"Generated {(i + 1) * stride_length} tokens in {time.time() - start:.2f} seconds")
                        # it/s
                        print(
                            f"{(i + 1) * stride_length / (time.time() - start):.2f} it/s")
                    break

            # print(kv_cache.get_seq_length().unsqueeze(0))
            torch.compiler.cudagraph_mark_step_begin()
            bucket_size = 250
            max_position = get_next_bucket(
                i + seq_len, bucket_size, TOKEN_LIMIT) if generate_token_backend == "cudagraphs-manual" else None
            outputs = generate_token(
                self._speech_embedding_cache,
                output_logits,
                i_tensor,
                batch_idx,
                self._speech_pos_embedding_cache,
                generated_ids,
                cfg_weight,
                temperature,
                self.repetition_penalty_processor,
                self.min_p_warper,
                self.top_p_warper,
                self.patched_model,
                kv_cache,
                stride_length,
                max_position=max_position,
            )
            output_logits = outputs[1]
            if len(outputs) == 3:
                generated_ids = outputs[2].clone()
            output_logits = output_logits.clone()

            if i == max_new_tokens // stride_length - 1:
                if benchmark:
                    torch.cuda.synchronize()  # For benchmarking to have correct it/s
                    print(
                        f"Stopping at {(i + 1) * stride_length} because max_new_tokens reached")
                    print(
                        f"Generated {(i + 1) * stride_length} tokens in {time.time() - start:.2f} seconds")
                    print(
                        f"{(i + 1) * stride_length / (time.time() - start):.2f} it/s")
        if benchmark:
            _t3_loop_end = time.time()
            print(f"[T3_TIMING] Token generation loop completed in {_t3_loop_end - _t3_loop_start:.3f}s", flush=True)

        return generated_ids

    @torch.inference_mode()
    def inference_turbo(self, t3_cond, text_tokens, temperature=0.8, top_k=1000, top_p=0.95, min_p=0.0, repetition_penalty=1.2,
                        max_gen_len=1000, disable_tqdm=False,
                        # Optimization parameters
                        generate_token_backend="reduce-overhead",
                        stride_length=4,
                        skip_when_1=True,
                        benchmark=False):
        """
        Inference method for the Turbo model (GPT2 backbone).
        Uses a simpler generation loop without CFG support.

        Optimizations applied:
        - Pre-allocated generated_ids tensor instead of list appending
        - In-place tensor updates with index_put_
        - Cached stop token tensor
        - Speech embedding cache for fast token-to-embedding lookup
        - Fast min-p and top-p warpers (CUDA graph compatible)
        - StaticCache for fixed memory KV cache
        - CUDA graphs with bucketing for token generation
        - Strided generation for reduced kernel launch overhead

        Args:
            t3_cond: T3Cond conditioning object
            text_tokens: Input text tokens
            temperature: Sampling temperature
            top_p: Top-p (nucleus) filtering  
            min_p: Min-p filtering threshold
            repetition_penalty: Repetition penalty
            max_gen_len: Maximum generation length
            disable_tqdm: Whether to disable progress bar (default False)
            generate_token_backend: Backend for token generation ("eager", "cudagraphs", "inductor", "cudagraphs-manual", or strided variants)
            stride_length: Number of tokens to generate per iteration (used with strided backends)
            skip_when_1: Skip top-p filtering when top_p=1.0 (default True)
            benchmark: Whether to print timing information (default False)
        """
        import time
        import sys

        device = self.device
        dtype = self.dtype

        # Initialize speech embedding cache for fast lookup (reuse across calls)
        self.init_speech_embedding_cache(
            vocab_size=self.hp.speech_tokens_dict_size, dtype=dtype)

        # Use fast warpers (CUDA graph compatible) instead of LogitsProcessorList
        self.update_processors(
            top_p, min_p, repetition_penalty, skip_when_1=skip_when_1, top_k=top_k)

        speech_start_token = self.hp.start_speech_token * \
            torch.ones_like(text_tokens[:, :1])
        embeds, _ = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_start_token,
            cfg_weight=0.0,
        )

        # Pre-allocate generated_ids tensor for efficiency (avoid list appending + torch.cat)
        # Use stop_speech_token as PAD since it's a valid index and won't affect generation
        # Must be within vocab range for repetition penalty
        PAD_TOKEN = self.hp.stop_speech_token
        generated_ids = torch.full(
            (1, max_gen_len + 1), PAD_TOKEN, dtype=torch.long, device=device)

        # Cache stop token tensor for comparison
        stop_token_tensor = torch.tensor(
            self.hp.stop_speech_token, device=device)

        # Batch index for index_put_
        batch_idx = torch.zeros(1, dtype=torch.long, device=device)

        # Get or create static KV cache for GPT2
        kv_cache = self._get_turbo_cache(
            max_batch_size=1,
            max_cache_len=embeds.size(1) + max_gen_len + 10,
            device=device,
            dtype=dtype,
        )

        # Initial forward pass with static cache
        seq_len = embeds.size(1)
        cache_position = torch.arange(seq_len, device=device)

        if benchmark:
            _t3_init_start = time.time()
            print(f"[TURBO_TIMING] Starting initial forward pass", flush=True)

        llm_outputs = self.tfmr(
            inputs_embeds=embeds,
            past_key_values=kv_cache,
            cache_position=cache_position,
            use_cache=True
        )

        hidden_states = llm_outputs[0]
        output_logits = self.speech_head(hidden_states[:, -1:, :]).clone()

        if benchmark:
            _t3_init_end = time.time()
            print(f"[TURBO_TIMING] Initial forward pass completed in {_t3_init_end - _t3_init_start:.3f}s", flush=True)

        # Setup indices for the generation loop
        indices = torch.arange(0, max_gen_len + 1, device=device)

        # Setup CUDA graphs wrapper if using cudagraphs-manual backend
        if generate_token_backend == "cudagraphs-manual":
            if not hasattr(self, "turbo_cudagraph_wrapper"):
                if benchmark:
                    print(f"[TURBO_TIMING] Creating new CUDA graph wrapper for turbo", flush=True)
                self.turbo_cudagraph_wrapper = T3TurboStepCUDAGraphWrapper(
                    generate_turbo_token,
                    self.tfmr,
                    self.speech_head,
                    kv_cache,
                    self.repetition_penalty_processor,
                    self.top_k_warper,
                    self.min_p_warper,
                    self.top_p_warper,
                    device=self.device,
                )
            else:
                # Must recapture graph for new generation (KV cache state is baked in)
                self.turbo_cudagraph_wrapper.mark_new_generation()
            self.turbo_cudagraph_wrapper.kv_cache = kv_cache  # Update cache reference
            _generate_turbo_token_variants["cudagraphs-manual"] = self.turbo_cudagraph_wrapper

        generate_token = _generate_turbo_token_variants.get(
            generate_token_backend, _generate_turbo_token_variants["eager"])

        if benchmark:
            print(
                f"[TURBO_TIMING] Starting token generation loop with backend: {generate_token_backend}, max_tokens: {max_gen_len}", flush=True)
            _t3_loop_start = time.time()
            torch.cuda.synchronize()

        # Use stride_length only for strided backends
        effective_stride = stride_length if "stride" in generate_token_backend else 1
        length_guesstimate = text_tokens.shape[1] * 2
        num_generated = 0  # Track actual generated tokens for EOS check

        for i in tqdm(range(max_gen_len // effective_stride), desc="Turbo Sampling", disable=disable_tqdm):
            i_tensor = indices[i * effective_stride]

            # Check for EOS token periodically (after estimated length)
            # Only check actually generated tokens, not the padded portion
            if num_generated > length_guesstimate and i % (20 // effective_stride) == 0:
                if (generated_ids[0, :num_generated] == stop_token_tensor).any():
                    if benchmark:
                        torch.cuda.synchronize()
                        print(
                            f"Stopping at {num_generated} because EOS token was generated")
                        print(
                            f"Generated {num_generated} tokens in {time.time() - _t3_loop_start:.2f} seconds")
                        print(
                            f"{num_generated / (time.time() - _t3_loop_start):.2f} it/s")
                    break

            torch.compiler.cudagraph_mark_step_begin()
            bucket_size = 250
            max_position = get_next_bucket(
                i + seq_len, bucket_size, TOKEN_LIMIT) if generate_token_backend == "cudagraphs-manual" else None

            outputs = generate_token(
                self._speech_embedding_cache,
                output_logits,
                i_tensor,
                batch_idx,
                generated_ids,
                temperature,
                self.repetition_penalty_processor,
                self.top_k_warper,
                self.min_p_warper,
                self.top_p_warper,
                self.tfmr,
                self.speech_head,
                kv_cache,
                seq_len,
                effective_stride,
                max_position=max_position,
            )
            output_logits = outputs[1]
            if len(outputs) == 3:
                generated_ids = outputs[2].clone()
            output_logits = output_logits.clone()
            num_generated += effective_stride  # Track generated tokens

            if i == max_gen_len // effective_stride - 1:
                if benchmark:
                    torch.cuda.synchronize()
                    print(
                        f"Stopping at {num_generated} because max_gen_len reached")
                    print(
                        f"Generated {num_generated} tokens in {time.time() - _t3_loop_start:.2f} seconds")
                    print(
                        f"{num_generated / (time.time() - _t3_loop_start):.2f} it/s")

        if benchmark:
            _t3_loop_end = time.time()
            print(
                f"[TURBO_TIMING] Token generation loop completed in {_t3_loop_end - _t3_loop_start:.3f}s", flush=True)

        return generated_ids

    @torch.inference_mode()
    def inference_turbo_streaming(
        self, 
        t3_cond, 
        text_tokens, 
        temperature=0.8, 
        top_k=1000, 
        top_p=0.95, 
        min_p=0.0, 
        repetition_penalty=1.2,
        max_gen_len=1000, 
        disable_tqdm=True,
        # Streaming parameters
        token_chunk_size=100,
        # Optimization parameters  
        generate_token_backend="eager",
        skip_when_1=True,
    ):
        """
        Streaming version of inference_turbo that yields speech tokens in chunks.
        
        This allows the caller to convert tokens to audio incrementally, reducing
        time-to-first-audio and enabling real-time playback.
        
        Args:
            token_chunk_size: Number of tokens to generate before yielding.
                             Lower = faster first chunk, Higher = more efficient.
            Other args: Same as inference_turbo
            
        Yields:
            torch.Tensor: Chunks of speech tokens (1D tensor)
        """
        device = self.device
        dtype = self.dtype

        # Initialize speech embedding cache
        self.init_speech_embedding_cache(
            vocab_size=self.hp.speech_tokens_dict_size, dtype=dtype)

        # Initialize processors
        self.update_processors(
            top_p, min_p, repetition_penalty, skip_when_1=skip_when_1, top_k=top_k)

        speech_start_token = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds, _ = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_start_token,
            cfg_weight=0.0,
        )

        # Pre-allocate generated_ids
        PAD_TOKEN = self.hp.stop_speech_token
        generated_ids = torch.full(
            (1, max_gen_len + 1), PAD_TOKEN, dtype=torch.long, device=device)

        stop_token_tensor = torch.tensor(self.hp.stop_speech_token, device=device)
        batch_idx = torch.zeros(1, dtype=torch.long, device=device)

        # Get KV cache
        kv_cache = self._get_turbo_cache(
            max_batch_size=1,
            max_cache_len=embeds.size(1) + max_gen_len + 10,
            device=device,
            dtype=dtype,
        )

        # Initial forward pass
        seq_len = embeds.size(1)
        cache_position = torch.arange(seq_len, device=device)

        llm_outputs = self.tfmr(
            inputs_embeds=embeds,
            past_key_values=kv_cache,
            cache_position=cache_position,
            use_cache=True
        )

        hidden_states = llm_outputs[0]
        output_logits = self.speech_head(hidden_states[:, -1:, :]).clone()

        indices = torch.arange(0, max_gen_len + 1, device=device)
        
        generate_token = _generate_turbo_token_variants.get(
            generate_token_backend, _generate_turbo_token_variants["eager"])

        num_generated = 0
        chunk_start = 0
        length_guesstimate = text_tokens.shape[1] * 2
        eos_found = False

        for i in tqdm(range(max_gen_len), desc="Streaming", disable=disable_tqdm):
            i_tensor = indices[i]

            # Check for EOS periodically
            if num_generated > length_guesstimate and i % 20 == 0:
                if (generated_ids[0, :num_generated] == stop_token_tensor).any():
                    eos_found = True
                    # Yield remaining tokens before stopping
                    if num_generated > chunk_start:
                        yield generated_ids[0, chunk_start:num_generated].clone()
                    break

            # Generate one token
            outputs = generate_token(
                self._speech_embedding_cache,
                output_logits,
                i_tensor,
                batch_idx,
                generated_ids,
                temperature,
                self.repetition_penalty_processor,
                self.top_k_warper,
                self.min_p_warper,
                self.top_p_warper,
                self.tfmr,
                self.speech_head,
                kv_cache,
                seq_len,
                1,  # stride_length=1 for streaming
                max_position=None,
            )
            output_logits = outputs[1].clone()
            if len(outputs) == 3:
                generated_ids = outputs[2].clone()
            num_generated += 1

            # Yield chunk when we have enough tokens
            if num_generated - chunk_start >= token_chunk_size:
                yield generated_ids[0, chunk_start:num_generated].clone()
                chunk_start = num_generated

        # Yield any remaining tokens
        if not eos_found and num_generated > chunk_start:
            yield generated_ids[0, chunk_start:num_generated].clone()

    def _get_turbo_cache(self, max_batch_size, max_cache_len, device, dtype):
        """Get or create a StaticCache for turbo inference."""
        if hasattr(self, '_turbo_cache'):
            params = self._turbo_cache_params
            if (params['max_batch_size'] == max_batch_size and
                params['max_cache_len'] >= max_cache_len and
                params['dtype'] == dtype and
                    params['device'] == device):
                # Reset and reuse existing cache
                self._turbo_cache.reset()
                return self._turbo_cache
            else:
                del self._turbo_cache

        cache = StaticCache(
            config=self.cfg,
            max_batch_size=max_batch_size,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
        )
        self._turbo_cache_params = {
            'max_batch_size': max_batch_size,
            'max_cache_len': max_cache_len,
            'device': device,
            'dtype': dtype,
        }
        self._turbo_cache = cache
        return cache


def _initial_forward_pass(
    inputs_embeds: Tensor,
    kv_cache: StaticCache,
    patched_model: T3HuggingfaceBackend,
    seq_len: int = 1,
):
    # trim padded inputs_embeds to the actual sequence length
    inputs_embeds = inputs_embeds[:, :seq_len, :]
    # Initial forward pass to get the logits for the first token
    cache_position = torch.arange(seq_len, device=inputs_embeds.device)
    output_logits = patched_model(
        inputs_embeds=inputs_embeds,
        past_key_values=kv_cache,
        cache_position=cache_position,
    )
    output_logits = output_logits[:, -1:, :]  # Normalize shape for loop
    return output_logits


_initial_forward_pass_variants = {
    "eager": _initial_forward_pass,
    "cudagraphs": torch.compile(_initial_forward_pass, backend="cudagraphs", fullgraph=True),
}


def generate_t3_token(
    _speech_embedding_cache: Tensor,
    output_logits: Tensor,
    i_tensor: Tensor,
    batch_idx: Tensor,
    position_embeds: Tensor,
    generated_ids: Tensor,
    cfg_weight: float,
    temperature: float,
    repetition_penalty_processor: RepetitionPenaltyLogitsProcessor,
    min_p_warper: MinPLogitsWarper,
    top_p_warper: TopPLogitsWarper,
    patched_model: T3HuggingfaceBackend,
    kv_cache: StaticCache,
    stride_length: int = 0,  # for API simplicity
    max_position: Optional[int] = None
):
    logits = output_logits[:, -1, :]

    # CFG
    if cfg_weight > 0.0:
        logits_cond = logits[0:1]
        logits_uncond = logits[1:2]
        logits = logits_cond + cfg_weight * (logits_cond - logits_uncond)

    logits = logits.squeeze(1)

    # Apply temperature scaling.
    if temperature != 1.0:
        logits = logits / temperature

    # Apply repetition penalty and top-p filtering.
    logits = repetition_penalty_processor(generated_ids, logits)
    logits = min_p_warper(None, logits)
    logits = top_p_warper(None, logits)

    # Convert logits to probabilities and sample the next token.
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)  # shape: (B, 1)

    # generated_ids[0, i + bos_len] = next_token.clone()
    generated_ids.index_put_((batch_idx, i_tensor), next_token.squeeze(-1))

    # Get embedding for the new token.
    # position_embed = position_embeds[i_tensor]
    position_embed = torch.index_select(
        position_embeds, 0, i_tensor).squeeze(0)
    next_token_embed = _speech_embedding_cache[next_token] + position_embed

    #  For CFG
    if cfg_weight > 0.0:
        next_token_embed = torch.cat([next_token_embed, next_token_embed])

    # max_position = kv_cache.get_seq_length().unsqueeze(0).item()

    return next_token, patched_model(
        inputs_embeds=next_token_embed,
        past_key_values=kv_cache,
        cache_position=kv_cache.get_seq_length().unsqueeze(0),
        max_position=max_position,
    )


def generate_t3_tokens_strided(
    _speech_embedding_cache: Tensor,
    output_logits: Tensor,
    i_tensor: Tensor,
    batch_idx: Tensor,
    position_embeds: Tensor,
    generated_ids: Tensor,
    cfg_weight: float,
    temperature: float,
    repetition_penalty_processor: RepetitionPenaltyLogitsProcessor,
    min_p_warper: MinPLogitsWarper,
    top_p_warper: TopPLogitsWarper,
    patched_model: T3HuggingfaceBackend,
    kv_cache: StaticCache,
    stride_length: int,
    max_position: Optional[int] = None
):
    for i in range(stride_length):
        next_token, output_logits = generate_t3_token(
            _speech_embedding_cache,
            output_logits,
            i_tensor.add(i),
            batch_idx,
            position_embeds,
            generated_ids,
            cfg_weight,
            temperature,
            repetition_penalty_processor,
            min_p_warper,
            top_p_warper,
            patched_model,
            kv_cache,
            stride_length,
        )
        output_logits = output_logits.clone()
    return next_token, output_logits


_generate_token_variants = {
    "eager": generate_t3_token,
    "cudagraphs": torch.compile(generate_t3_token, backend="cudagraphs", fullgraph=True),
    "inductor": torch.compile(generate_t3_token, backend="inductor", fullgraph=True, mode="max-autotune"),
    "cudagraphs-strided": torch.compile(generate_t3_tokens_strided, backend="cudagraphs", fullgraph=True),
    "inductor-strided": torch.compile(generate_t3_tokens_strided, backend="inductor", fullgraph=True, mode="max-autotune"),
}


# ============================================================================
# Turbo (GPT2) specific token generation functions
# ============================================================================

def generate_turbo_token(
    _speech_embedding_cache: Tensor,
    output_logits: Tensor,
    i_tensor: Tensor,
    batch_idx: Tensor,
    generated_ids: Tensor,
    temperature: float,
    repetition_penalty_processor: RepetitionPenaltyLogitsProcessor,
    top_k_warper,
    min_p_warper: MinPLogitsWarper,
    top_p_warper: TopPLogitsWarper,
    tfmr,  # GPT2Model
    speech_head: nn.Linear,
    kv_cache: StaticCache,
    seq_len: int,
    stride_length: int = 1,  # For API compatibility
    max_position: Optional[int] = None,
    cache_pos: Optional[Tensor] = None,  # External cache position for CUDA graphs
):
    """
    Single token generation step for Turbo (GPT2) model.
    Similar to generate_t3_token but uses GPT2 directly instead of patched LLaMA model.
    
    Args:
        cache_pos: Optional external cache position tensor. If provided, used directly
                   instead of querying kv_cache. Required for persistent CUDA graphs.
    """
    logits = output_logits[:, -1, :]

    # Apply temperature scaling
    if temperature != 1.0:
        logits = logits / temperature

    # Apply repetition penalty and filtering
    logits = repetition_penalty_processor(generated_ids, logits)
    logits = top_k_warper(None, logits)
    logits = min_p_warper(None, logits)
    logits = top_p_warper(None, logits)

    # Convert logits to probabilities and sample
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)  # shape: (B, 1)

    # Store in generated_ids
    generated_ids.index_put_((batch_idx, i_tensor), next_token.squeeze(-1))

    # Get embedding for the new token
    next_token_embed = _speech_embedding_cache[next_token]

    # Get cache position for the next forward pass
    # Use external cache_pos if provided (for persistent CUDA graphs)
    if cache_pos is None:
        cache_pos = kv_cache.get_seq_length().unsqueeze(0)

    # Forward pass through GPT2
    llm_outputs = tfmr(
        inputs_embeds=next_token_embed,
        past_key_values=kv_cache,
        cache_position=cache_pos,
        use_cache=True,
    )

    hidden_states = llm_outputs[0]
    next_logits = speech_head(hidden_states[:, -1:, :])

    return next_token, next_logits


def generate_turbo_tokens_strided(
    _speech_embedding_cache: Tensor,
    output_logits: Tensor,
    i_tensor: Tensor,
    batch_idx: Tensor,
    generated_ids: Tensor,
    temperature: float,
    repetition_penalty_processor: RepetitionPenaltyLogitsProcessor,
    top_k_warper,
    min_p_warper: MinPLogitsWarper,
    top_p_warper: TopPLogitsWarper,
    tfmr,
    speech_head: nn.Linear,
    kv_cache: StaticCache,
    seq_len: int,
    stride_length: int,
    max_position: Optional[int] = None,
):
    """Strided token generation for Turbo model."""
    for i in range(stride_length):
        next_token, output_logits = generate_turbo_token(
            _speech_embedding_cache,
            output_logits,
            i_tensor.add(i),
            batch_idx,
            generated_ids,
            temperature,
            repetition_penalty_processor,
            top_k_warper,
            min_p_warper,
            top_p_warper,
            tfmr,
            speech_head,
            kv_cache,
            seq_len,
            stride_length,
        )
        output_logits = output_logits.clone()
    return next_token, output_logits


_generate_turbo_token_variants = {
    "eager": generate_turbo_token,
    "cudagraphs": torch.compile(generate_turbo_token, backend="cudagraphs", fullgraph=True),
    "inductor": torch.compile(generate_turbo_token, backend="inductor", fullgraph=True, mode="max-autotune"),
    "cudagraphs-strided": torch.compile(generate_turbo_tokens_strided, backend="cudagraphs", fullgraph=True),
    "inductor-strided": torch.compile(generate_turbo_tokens_strided, backend="inductor", fullgraph=True, mode="max-autotune"),
    # reduce-overhead is recommended for real-time: automatic graph caching with minimal overhead
    "reduce-overhead": torch.compile(generate_turbo_token, mode="reduce-overhead"),
}


class T3TurboStepCUDAGraphWrapper:
    """
    Manual CUDA graph wrapper for Turbo (GPT2) model token generation.
    
    Note: This wrapper recaptures the graph for each new generation because
    CUDA graphs bake in the KV cache's internal state. For real-time applications
    where recapture overhead matters, consider using "reduce-overhead" backend instead.
    """

    def __init__(
        self,
        generate_token: Callable,
        tfmr,
        speech_head: nn.Linear,
        kv_cache: StaticCache,
        repetition_penalty_processor,
        top_k_warper,
        min_p_warper,
        top_p_warper,
        device: torch.device = None,
    ):
        self.generate_token = generate_token
        self.tfmr = tfmr
        self.speech_head = speech_head
        self.kv_cache = kv_cache
        self.repetition_penalty_processor = repetition_penalty_processor
        self.top_k_warper = top_k_warper
        self.min_p_warper = min_p_warper
        self.top_p_warper = top_p_warper
        self.device = device or next(tfmr.parameters()).device

        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_tensors: Optional[dict] = None
        self._captured = False

    def _capture_graph(
        self,
        speech_embedding_cache: torch.Tensor,
        output_logits: torch.Tensor,
        i_tensor: torch.Tensor,
        batch_idx: torch.Tensor,
        generated_ids: torch.Tensor,
        temperature: float,
        seq_len: int,
        stride_length: int,
        max_position: Optional[int] = None,
    ) -> None:
        """Capture CUDA graph for token generation."""
        self._graph = torch.cuda.CUDAGraph()
        static_tensors = {}

        # Clone all tensors for static graph capture
        static_tensors["speech_embedding_cache"] = speech_embedding_cache.clone()
        static_tensors["output_logits"] = output_logits.clone()
        static_tensors["i_tensor"] = i_tensor.clone()
        static_tensors["batch_idx"] = batch_idx.clone()
        static_tensors["generated_ids"] = generated_ids.clone()
        static_tensors["temperature"] = temperature
        static_tensors["seq_len"] = seq_len
        static_tensors["stride_length"] = stride_length
        static_tensors["max_position"] = max_position

        with torch.inference_mode():
            with torch.cuda.graph(self._graph):
                static_tensors["out_1"], static_tensors["out_2"] = self.generate_token(
                    static_tensors["speech_embedding_cache"],
                    static_tensors["output_logits"],
                    static_tensors["i_tensor"],
                    static_tensors["batch_idx"],
                    static_tensors["generated_ids"],
                    static_tensors["temperature"],
                    self.repetition_penalty_processor,
                    self.top_k_warper,
                    self.min_p_warper,
                    self.top_p_warper,
                    self.tfmr,
                    self.speech_head,
                    self.kv_cache,
                    static_tensors["seq_len"],
                    static_tensors["stride_length"],
                    static_tensors["max_position"],
                    None,  # cache_pos - let it compute internally
                )

        self._static_tensors = static_tensors
        self._captured = True

    def __call__(
        self,
        speech_embedding_cache: torch.Tensor,
        output_logits: torch.Tensor,
        i_tensor: torch.Tensor,
        batch_idx: torch.Tensor,
        generated_ids: torch.Tensor,
        temperature: float,
        repetition_penalty_processor=None,
        min_p_warper=None,
        top_p_warper=None,
        tfmr=None,
        speech_head=None,
        kv_cache=None,
        seq_len: int = 0,
        stride_length: int = 1,
        max_position: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if not self._captured:
            # First call - capture the graph
            self._capture_graph(
                speech_embedding_cache,
                output_logits,
                i_tensor,
                batch_idx,
                generated_ids,
                temperature,
                seq_len,
                stride_length,
                max_position,
            )
        else:
            # Subsequent calls - update static tensors and replay
            st = self._static_tensors
            st["speech_embedding_cache"].copy_(speech_embedding_cache)
            st["output_logits"].copy_(output_logits)
            st["i_tensor"].copy_(i_tensor)
            st["batch_idx"].copy_(batch_idx)
            st["generated_ids"].copy_(generated_ids)
            st["temperature"] = temperature
            st["seq_len"] = seq_len
            st["stride_length"] = stride_length
            st["max_position"] = max_position

            self._graph.replay()
        
        return (
            self._static_tensors["out_1"],
            self._static_tensors["out_2"],
            self._static_tensors["generated_ids"],
        )

    def reset(self) -> None:
        """Clear captured graph (forces recapture on next call)."""
        self._graph = None
        self._static_tensors = None
        self._captured = False

    def mark_new_generation(self):
        """
        Prepare for a new generation - must recapture graph because
        KV cache state is baked into the graph.
        """
        self.reset()
        self.kv_cache.reset()
