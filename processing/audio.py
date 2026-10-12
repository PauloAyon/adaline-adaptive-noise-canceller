from pathlib import Path

import numpy as np
from scipy.io import wavfile


def read_audio(file_path, seconds_to_process):
    # Require a readable, non-empty WAV with one or two channels
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    sample_rate, audio = wavfile.read(file_path)
    if audio.size == 0:
        raise ValueError("The input audio is empty.")
    if audio.ndim not in (1, 2):
        raise ValueError("The input audio must have one or two dimensions.")

    # Convert integer PCM samples to a floating-point range near -1 to 1
    if np.issubdtype(audio.dtype, np.integer):
        limits = np.iinfo(audio.dtype)
        if np.issubdtype(audio.dtype, np.unsignedinteger):
            midpoint = (int(limits.max) + 1) / 2
            audio = (audio.astype(np.float64) - midpoint) / midpoint
        else:
            scale = max(abs(int(limits.min)), abs(int(limits.max)))
            audio = audio.astype(np.float64) / scale
    else:
        audio = audio.astype(np.float64)

    if not np.all(np.isfinite(audio)):
        raise ValueError("The input audio contains non-finite values.")

    # Average stereo channels to produce one mono signal
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Keep only the requested number of seconds
    sample_count = int(seconds_to_process * sample_rate)
    if sample_count < 1:
        raise ValueError("The processing duration must be greater than zero.")
    return sample_rate, audio[:sample_count]


def save_audio(file_path, sample_rate, signal):
    # Create the destination folder before writing the WAV
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    signal = np.asarray(signal, dtype=np.float64)
    if signal.size == 0 or not np.all(np.isfinite(signal)):
        raise ValueError("Cannot save an empty or non-finite audio signal.")

    # Scale each signal to fit the signed 16-bit WAV range
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak

    audio_int16 = np.clip(signal * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(file_path, sample_rate, audio_int16)