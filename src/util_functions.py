import datetime
import functools
import hashlib
import os
import time
from pathlib import Path

import psutil
import torchaudio


@functools.cache
def get_process_creation_time():
    """Get the process creation time as a datetime object"""
    p = psutil.Process(os.getpid())
    creation_timestamp = p.create_time()
    return datetime.datetime.fromtimestamp(creation_timestamp)

@functools.cache
def get_cache_dir():
    """Get or create the conditionals cache directory"""
    cache_dir = Path("cache/conditionals")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

@functools.cache
def get_cache_key(audio_path, uuid, exaggeration=None):
    """Generate a cache key based on audio file, UUID, and exaggeration"""
    if audio_path is None:
        return None

    # Extract just the filename without extension as prefix
    try:
        filename = Path(audio_path).stem  # Gets filename without extension
        # Remove any temp directory prefixes, just keep the actual filename
        cache_prefix = filename
    except Exception:
        cache_prefix = "unknown"

    # Convert UUID to hex string for readability
    try:
        uuid_hex = hex(uuid)[2:]  # Remove '0x' prefix
    except (TypeError, ValueError):
        uuid_hex = str(uuid)

    # Create cache key: prefix_uuid_exaggeration
    if exaggeration is None:
        cache_key = f"{cache_prefix}_{uuid_hex}"
    else:
        cache_key = f"{cache_prefix}_{uuid_hex}_{exaggeration:.2f}"

    # Use MD5 hash if the key gets too long (over 100 chars)
    if len(cache_key) > 100:
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return f"{cache_prefix}_{cache_hash}"

    return cache_key

@functools.cache
def get_wavout_dir():
    formatted_start_time = get_process_creation_time().strftime("%Y%m%d_%H%M%S")
    wavout_dir = Path("output_temp").joinpath(formatted_start_time)
    wavout_dir.mkdir(parents=True, exist_ok=True)
    return wavout_dir

def save_torchaudio_wav(wav_tensor, sr, audio_path, uuid):
    """Save a tensor as a WAV file using torchaudio"""

    formatted_now_time = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")

    filename = f"{formatted_now_time}_{get_cache_key(audio_path, uuid)}"
    path = get_wavout_dir() / f"{filename}.wav"
    torchaudio.save(path, wav_tensor, sr, encoding="PCM_S", bits_per_sample=16)
    return str(path.resolve())