"""
Model loading and management utilities for Zonos application.
"""

import sys
import torch
from loguru import logger
from typing import Optional, TYPE_CHECKING, Union

try:
    from cache_utils import init_conditional_memory_cache, clear_memory_cache
except ImportError:
    from .cache_utils import init_conditional_memory_cache, clear_memory_cache

if TYPE_CHECKING:
    try:
        from chatterbox.tts import ChatterboxTTS
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    except ImportError:
        from .chatterbox.tts import ChatterboxTTS
        from .chatterbox.tts_turbo import ChatterboxTurboTTS
        from .chatterbox.mtl_tts import ChatterboxMultilingualTTS


CURRENT_MODEL_TYPE: Optional[str] = None
CURRENT_MODEL: Union["ChatterboxTTS","ChatterboxTurboTTS","ChatterboxMultilingualTTS", None] = None

def safe_conditional_to_dtype(model, dtype: torch.dtype) -> None:
    """
    Safely move model conditionals to the specified dtype with comprehensive null checks.
    
    Args:
        model: The model containing conditionals
        dtype: Target torch.dtype
    """
    if hasattr(model, 'conds') and model.conds is not None:
        if hasattr(model.conds, 't3') and model.conds.t3 is not None:
            model.conds.t3.to(dtype=dtype)

def initialize_model_dtype(model, dtype: torch.dtype) -> None:
    """
    Initialize a Chatterbox model with proper dtype handling for all components.
    
    Args:
        model: ChatterboxTTS or ChatterboxMultilingualTTS instance
        dtype: Target torch.dtype
    """
    # Move main t3 model
    if hasattr(model, 't3') and model.t3 is not None:
        model.t3.to(dtype=dtype)
    
    # Move conditionals if they exist
    safe_conditional_to_dtype(model, dtype)

def load_model_if_needed(model_choice: str,
                        device: torch.device,
                        dtype: torch.dtype,
                        supported_languages: list[str]) -> Union["ChatterboxTTS","ChatterboxTurboTTS","ChatterboxMultilingualTTS", None]:

    global CURRENT_MODEL_TYPE, CURRENT_MODEL

    if CURRENT_MODEL_TYPE != model_choice:
        logger.info(f"Model type changed from {CURRENT_MODEL_TYPE} to {model_choice}. Reloading model...")
        if CURRENT_MODEL is not None:
            clear_memory_cache()
            del CURRENT_MODEL
            torch.cuda.empty_cache()

        logger.info(f"Loading {model_choice} model...")
       
        try:
            if model_choice == "TURBO":
                #logger.info("Loading Turbo Model")
                try:
                    from chatterbox.tts_turbo import ChatterboxTurboTTS as Chatterbox
                except ImportError:
                    from .chatterbox.tts_turbo import ChatterboxTurboTTS as Chatterbox
            elif model_choice == "MULTILINGUAL":
                #logger.info("Loading Multilingual Model")
                try:
                    from chatterbox.mtl_tts import ChatterboxMultilingualTTS as Chatterbox
                except ImportError:
                    from .chatterbox.mtl_tts import ChatterboxMultilingualTTS as Chatterbox
            else:
                #logger.info("Loading English Model")
                try:
                    from chatterbox.tts import ChatterboxTTS as Chatterbox
                except ImportError:
                    from .chatterbox.tts import ChatterboxTTS as Chatterbox
            model = Chatterbox.from_pretrained(device)
            initialize_model_dtype(model, dtype)
            torch.compiler.reset()
        except Exception as e:
            logger.error(f"Error while loading {model_choice} model: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        logger.info(f"{model_choice} model loaded successfully!")

        init_conditional_memory_cache(model, device, dtype, supported_languages=supported_languages)

        CURRENT_MODEL = model
        CURRENT_MODEL_TYPE = model_choice   

    return CURRENT_MODEL

