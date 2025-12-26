from dataclasses import dataclass
from pathlib import Path

import torch

from .models.t3.modules.cond_enc import T3Cond


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def clone(self):
        """Create a deep copy of the Conditionals object with proper tensor cloning"""
        # Clone T3Cond - create new instance with cloned tensors
        t3_clone = T3Cond(
            speaker_emb=self.t3.speaker_emb.clone().detach() if self.t3.speaker_emb is not None else None,
            clap_emb=self.t3.clap_emb.clone().detach() if hasattr(self.t3, 'clap_emb') and self.t3.clap_emb is not None else None,
            cond_prompt_speech_tokens=self.t3.cond_prompt_speech_tokens.clone().detach() if self.t3.cond_prompt_speech_tokens is not None else None,
            cond_prompt_speech_emb=self.t3.cond_prompt_speech_emb.clone().detach() if hasattr(self.t3, 'cond_prompt_speech_emb') and self.t3.cond_prompt_speech_emb is not None else None,
            emotion_adv=self.t3.emotion_adv.clone().detach() if self.t3.emotion_adv is not None else None
        )
        
        # Clone gen dict with proper tensor handling
        gen_clone = {}
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                gen_clone[k] = v.clone().detach()
            elif isinstance(v, (list, tuple)):
                # Handle list/tuple of tensors
                gen_clone[k] = type(v)(
                    item.clone().detach() if torch.is_tensor(item) else item 
                    for item in v
                )
            else:
                # For non-tensor values, use regular copy
                gen_clone[k] = v
        
        return Conditionals(t3=t3_clone, gen=gen_clone)

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])
