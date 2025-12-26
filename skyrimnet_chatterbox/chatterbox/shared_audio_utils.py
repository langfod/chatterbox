import math
import wave
import struct

import warnings

from pathlib import Path
from typing import Union, Tuple

import torch
import torchaudio
import torchaudio.functional as taF

from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR


def _biquad_filter_torch(x: torch.Tensor, b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """
    Apply a biquad IIR filter using Direct Form II Transposed.
    Pure PyTorch implementation of scipy.signal.lfilter for biquad (2nd order) filters.

    Args:
        x: Input signal tensor (1D)
        b: Numerator coefficients [b0, b1, b2]
        a: Denominator coefficients [a0, a1, a2] (a0 should be 1.0)

    Returns:
        Filtered signal tensor
    """
    # Ensure float64 for numerical stability
    x = x.double()
    b = b.double()
    a = a.double()

    # Initialize output and state variables
    y = torch.zeros_like(x)
    z1 = torch.tensor(0.0, dtype=torch.float64, device=x.device)
    z2 = torch.tensor(0.0, dtype=torch.float64, device=x.device)

    b0, b1, b2 = b[0], b[1], b[2]
    a1, a2 = a[1], a[2]

    # Direct Form II Transposed
    for n in range(len(x)):
        xn = x[n]
        yn = b0 * xn + z1
        z1 = b1 * xn - a1 * yn + z2
        z2 = b2 * xn - a2 * yn
        y[n] = yn

    return y.float()


def _compute_k_weighting_coeffs(sr: float, device: torch.device) -> tuple:
    """
    Compute K-weighting filter coefficients for ITU-R BS.1770.
    Returns coefficients for high shelf and high pass biquad filters.
    """
    # High shelf filter: G=4dB, Q=1/sqrt(2), fc=1500Hz
    G_hs = 4.0
    Q_hs = 1.0 / math.sqrt(2)
    fc_hs = 1500.0

    A = 10.0 ** (G_hs / 40.0)
    w0 = 2.0 * math.pi * fc_hs / sr
    alpha = math.sin(w0) / (2.0 * Q_hs)

    cos_w0 = math.cos(w0)
    sqrt_A = math.sqrt(A)

    # High shelf coefficients
    b0_hs = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
    b1_hs = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2_hs = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
    a0_hs = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
    a1_hs = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2_hs = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha

    b_hs = torch.tensor([b0_hs / a0_hs, b1_hs / a0_hs,
                        b2_hs / a0_hs], device=device)
    a_hs = torch.tensor([1.0, a1_hs / a0_hs, a2_hs / a0_hs], device=device)

    # High pass filter: G=0dB, Q=0.5, fc=38Hz
    Q_hp = 0.5
    fc_hp = 38.0

    w0_hp = 2.0 * math.pi * fc_hp / sr
    alpha_hp = math.sin(w0_hp) / (2.0 * Q_hp)
    cos_w0_hp = math.cos(w0_hp)

    # High pass coefficients
    b0_hp = (1 + cos_w0_hp) / 2
    b1_hp = -(1 + cos_w0_hp)
    b2_hp = (1 + cos_w0_hp) / 2
    a0_hp = 1 + alpha_hp
    a1_hp = -2 * cos_w0_hp
    a2_hp = 1 - alpha_hp

    b_hp = torch.tensor([b0_hp / a0_hp, b1_hp / a0_hp,
                        b2_hp / a0_hp], device=device)
    a_hp = torch.tensor([1.0, a1_hp / a0_hp, a2_hp / a0_hp], device=device)

    return (b_hs, a_hs), (b_hp, a_hp)


def compute_integrated_loudness_torch(wav: torch.Tensor, sr: int, device: torch.device = None) -> float:
    """
    Compute integrated loudness (LUFS) using ITU-R BS.1770-4 algorithm.
    Pure PyTorch implementation for mono audio.

    Args:
        wav: Audio tensor (1D mono)
        sr: Sample rate in Hz
        device: Torch device

    Returns:
        Integrated loudness in LUFS
    """
    if device is None:
        device = wav.device

    # Ensure 1D mono
    wav = wav.flatten()
    num_samples = wav.shape[0]

    # Get K-weighting filter coefficients
    (b_hs, a_hs), (b_hp, a_hp) = _compute_k_weighting_coeffs(sr, device)

    # Apply K-weighting filters: high shelf then high pass
    filtered = _biquad_filter_torch(wav, b_hs, a_hs)
    filtered = _biquad_filter_torch(filtered, b_hp, a_hp)

    # Gating parameters
    T_g = 0.4  # 400ms gating block
    Gamma_a = -70.0  # Absolute threshold in LKFS
    overlap = 0.75
    step = 1.0 - overlap

    T = num_samples / sr  # Length in seconds
    num_blocks = int(round((T - T_g) / (T_g * step)) + 1)

    if num_blocks <= 0:
        # Audio too short for gating, use simple mean square
        mean_sq = (filtered ** 2).mean().item()
        if mean_sq > 0:
            return -0.691 + 10.0 * math.log10(mean_sq)
        return -float('inf')

    # Calculate mean square for each block
    z = torch.zeros(num_blocks, device=device, dtype=torch.float64)

    for j in range(num_blocks):
        l_idx = int(T_g * (j * step) * sr)
        u_idx = int(T_g * (j * step + 1) * sr)
        u_idx = min(u_idx, num_samples)
        block_data = filtered[l_idx:u_idx].double()
        z[j] = (block_data ** 2).sum() / (T_g * sr)

    # Calculate loudness for each block
    l_j = torch.zeros(num_blocks, device=device, dtype=torch.float64)
    for j in range(num_blocks):
        if z[j] > 0:
            l_j[j] = -0.691 + 10.0 * math.log10(z[j].item())
        else:
            l_j[j] = -float('inf')

    # First gate: absolute threshold
    J_g = [j for j in range(num_blocks) if l_j[j].item() >= Gamma_a]

    if not J_g:
        return -float('inf')

    # Calculate average z for gated blocks
    z_avg_gated = sum(z[j].item() for j in J_g) / len(J_g)

    # Calculate relative threshold
    if z_avg_gated > 0:
        Gamma_r = -0.691 + 10.0 * math.log10(z_avg_gated) - 10.0
    else:
        return -float('inf')

    # Second gate: relative and absolute thresholds
    J_g = [j for j in range(num_blocks) if l_j[j].item() > Gamma_r and l_j[j].item() > Gamma_a]

    if not J_g:
        return -float('inf')

    # Final average
    z_avg_final = sum(z[j].item() for j in J_g) / len(J_g)

    # Calculate final LUFS
    if z_avg_final > 0:
        return -0.691 + 10.0 * math.log10(z_avg_final)
    return -float('inf')


def norm_loudness(wav: torch.Tensor, sr: int, target_lufs: float = -27.0) -> torch.Tensor:
    """
    Normalize mono audio to target LUFS using pure PyTorch implementation.
    Implements ITU-R BS.1770-4 loudness measurement.
    
    Args:
        wav: Mono audio tensor (1D)
        sr: Sample rate in Hz
        target_lufs: Target loudness in LUFS (default: -27)
    
    Returns:
        Loudness-normalized audio tensor
    """
    try:
        loudness = compute_integrated_loudness_torch(wav, sr, device=wav.device)

        if not math.isfinite(loudness):
            return wav

        gain_db = target_lufs - loudness
        gain_linear = 10.0 ** (gain_db / 20.0)

        if math.isfinite(gain_linear) and gain_linear > 0.0:
            wav = wav * gain_linear
    except Exception as e:
        print(f"Warning: Error in norm_loudness, skipping: {e}")
    return wav


def save_tensor_as_wav(tensor_data, filename, sample_rate=24000, n_channels=1, sampwidth=2):
    """
    Saves a PyTorch tensor as a WAV file.

    Args:
        tensor_data (torch.Tensor): The audio data as a PyTorch tensor.
                                     Expected to be a 1D tensor for mono,
                                     or a 2D tensor with shape [num_samples, num_channels] for multi-channel.
        filename (str): The name of the output WAV file.
        sample_rate (int): The sample rate of the audio.
        n_channels (int): The number of audio channels.
        sampwidth (int): The sample width in bytes (e.g., 2 for 16-bit PCM).
    """
    # Convert tensor to appropriate data type and scale if necessary
    # For 16-bit PCM (sampwidth=2), values typically range from -32768 to 32767
    if sampwidth == 2:
        # Assuming float tensor in range [-1, 1], scale to 16-bit PCM range
        # Clamp to avoid overflow when converting to int16
        scaled_data = (tensor_data * 32767).clamp(-32768, 32767).to(torch.int16)
    else:
        raise ValueError("Unsupported sample width. Only 2 bytes (16-bit PCM) is implemented.")

    # Flatten the tensor data for writing (still on GPU if input was on GPU)
    if n_channels > 1:
        # Interleave channels if multi-channel
        scaled_data = scaled_data.transpose(0, 1).contiguous().view(-1)
    else:
        scaled_data = scaled_data.view(-1)

    # .contiguous() ensures memory layout is correct, then get underlying storage as bytes
    if scaled_data.is_cuda:
        scaled_data = scaled_data.cpu()
    
    # Use struct.pack with format for entire array at once
    # '<' = little-endian, 'h' = signed short (2 bytes), repeated for all samples
    num_samples = scaled_data.numel()
    packed_data = struct.pack(f'<{num_samples}h', *scaled_data.tolist())
    #print(f"Saving WAV file: {filename}, Sample Rate: {sample_rate}, Channels: {n_channels}, Samples: {num_samples}")
    with wave.open(str(filename), 'wb') as wav_file:
        wav_file.setnchannels(n_channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(packed_data)

def load_wav_as_tensor(file_path, normalize=True, mono=True) -> Tuple[torch.Tensor, int]:
    """
    Loads a WAV file and returns its audio data as a PyTorch tensor.
    
    Args:
        file_path: Path to the WAV file
        normalize: If True, returns float32 tensor normalized to [-1, 1] range.
                   If False, returns original integer dtype tensor.
    
    Returns:
        tuple: (audio_tensor, framerate)
            - audio_tensor: Shape (n_channels, n_frames), dtype float32 if normalize=True
            - framerate: Sample rate of the audio
    """
    with wave.open(str(file_path), 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        
        # Read all audio frames as raw bytes
        # Note: Some WAV files have invalid nframes (-1 or very large values),
        # so we read all frames and calculate actual count from bytes read
        frames_bytes = wf.readframes(wf.getnframes())
        
        # Calculate actual frame count from bytes read (more reliable than getnframes)
        n_frames = len(frames_bytes) // (n_channels * sampwidth)

        # Determine the format string and normalization factor based on sample width
        if sampwidth == 1:  # 8-bit unsigned
            format_string = f'{n_frames * n_channels}B'
            dtype = torch.uint8
            max_val = 255.0
            offset = 128  # 8-bit audio is unsigned, center is 128
        elif sampwidth == 2:  # 16-bit signed
            format_string = f'<{n_frames * n_channels}h'
            dtype = torch.int16
            max_val = 32768.0
            offset = 0
        elif sampwidth == 3:  # 24-bit signed
            # 24-bit requires manual unpacking (3 bytes per sample)
            audio_data_list = []
            for i in range(0, len(frames_bytes), 3):
                # Little-endian: sign-extend 24-bit to 32-bit
                sample = frames_bytes[i] | (frames_bytes[i+1] << 8) | (frames_bytes[i+2] << 16)
                # Sign extend if negative (bit 23 is set)
                if sample & 0x800000:
                    sample -= 0x1000000
                audio_data_list.append(sample)
            dtype = torch.int32
            max_val = 8388608.0  # 2^23
            offset = 0
        elif sampwidth == 4:  # 32-bit signed
            format_string = f'<{n_frames * n_channels}i'
            dtype = torch.int32
            max_val = 2147483648.0  # 2^31
            offset = 0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

        # Unpack raw bytes into a list of integers (except for 24-bit which is already done)
        if sampwidth != 3:
            audio_data_list = list(struct.unpack(format_string, frames_bytes))

        # Convert to tensor
        audio_tensor = torch.tensor(audio_data_list, dtype=dtype)

        # Reshape to (n_channels, n_frames)
        # WAV files store interleaved samples (L, R, L, R, ...), so we need to
        # reshape to (n_frames, n_channels) first, then transpose
        audio_tensor = audio_tensor.reshape(n_frames, n_channels).T

        # Normalize to float32 in range [-1, 1] if requested
        if normalize:
            audio_tensor = audio_tensor.to(torch.float32)
            if offset != 0:
                audio_tensor = (audio_tensor - offset) / (max_val / 2)
            else:
                audio_tensor = audio_tensor / max_val

        # Convert to mono by averaging channels if requested
        if mono and audio_tensor.shape[0] > 1:
            audio_tensor = audio_tensor.mean(dim=0, keepdim=True)

        return audio_tensor, framerate


def load_and_preprocess_audio(
    wav_fpath: Union[str, Path],
    device: str,
    min_duration: float = 0.1,
    normalize: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load, validate, and preprocess audio for TTS conditioning.

    Args:
        wav_fpath: Path to the audio file
        device: Target device for tensors
        min_duration: Minimum required audio duration in seconds
        normalize: Whether to apply loudness normalization

    Returns:
        Tuple of (s3gen_ref_wav, ref_16k_wav_tensor) - both preprocessed and on device

    Raises:
        RuntimeError: If audio loading or resampling fails
        ValueError: If audio is invalid or too short
    """
    # Load reference wav with enhanced error handling
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            #s3gen_ref_wav, _sr = torchaudio.load(wav_fpath)
            s3gen_ref_wav, _sr = load_wav_as_tensor(file_path=wav_fpath, normalize=True, mono=True)
            #print(f"Loaded audio: {wav_fpath}, shape: {s3gen_ref_wav.shape}, sample rate: {_sr}")
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file '{wav_fpath}': {e}")

    # Validate audio data
    if s3gen_ref_wav.numel() == 0:
        raise ValueError("Audio file is empty or contains no valid audio data")

    if _sr <= 0:
        raise ValueError(f"Invalid sample rate: {_sr}")

    # Convert to mono FIRST (before any resampling for efficiency)
    if s3gen_ref_wav.dim() > 1 and s3gen_ref_wav.shape[0] > 1:
        # Average channels to mono
        s3gen_ref_wav = s3gen_ref_wav.mean(dim=0)
    else:
        # Already mono or single channel, just flatten
        s3gen_ref_wav = s3gen_ref_wav.flatten()

    # Check audio duration
    duration = s3gen_ref_wav.shape[0] / _sr
    if duration < min_duration:
        raise ValueError(
            f"Audio too short: {duration:.2f}s (minimum {min_duration}s required)")

    # Normalize loudness at original sample rate (most accurate)
    if normalize:
        s3gen_ref_wav = norm_loudness(s3gen_ref_wav, _sr)

    # Resample to S3GEN_SR if necessary
    if _sr != S3GEN_SR:
        try:
            # Add batch dim for resample, then remove
            s3gen_ref_wav = taF.resample(s3gen_ref_wav.unsqueeze(0), _sr, S3GEN_SR).squeeze(0)
        except Exception as e:
            raise RuntimeError(
                f"Failed to resample audio from {_sr}Hz to {S3GEN_SR}Hz: {e}")

    # Move to device
    s3gen_ref_wav = s3gen_ref_wav.to(device)

    # Resample to 16kHz for encoder
    try:
        ref_16k_wav_tensor = taF.resample(
            s3gen_ref_wav.unsqueeze(0), S3GEN_SR, S3_SR).squeeze(0)
    except Exception as e:
        raise RuntimeError(f"Failed to resample to 16kHz: {e}")

    return s3gen_ref_wav, ref_16k_wav_tensor
