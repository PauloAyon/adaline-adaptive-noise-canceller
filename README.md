# ADALINE Adaptive Noise Canceller

This project simulates environmental noise over an audio signal and uses an ADALINE filter with the LMS learning rule to estimate that noise and recover the original audio.

## Requirements

- Python 3.10 or newer
- A WAV audio file that you are allowed to use

## Input Audio

Add your own WAV file to the project folder. The program expects the filename `blackMagicWoman_WAV.wav`, so you can either rename your file to match or change `AUDIO_FILE` near the top of `main.py` to your filename, for example `AUDIO_FILE = PROJECT_DIR / "my_song.wav"`. The default filename is ignored by Git and is not included in the repository. If you use a different filename, add it to `.gitignore` as well.

## Installation

Open PowerShell in the project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

Run the full simulation and save plots and audio without opening plot windows:

```powershell
python main.py
```

Skip plot generation to reduce processing and memory use:

```powershell
python main.py --no-plots
```

Show plots after saving them:

```powershell
python main.py --show-plots
```

Set a seed to make the generated noise repeatable:

```powershell
python main.py --seed 42
```

Audio playback is optional. Install its dependency and add `--play`:

```powershell
python -m pip install -r requirements-audio.txt
python main.py --play
```

## Parameters

Edit the values near the top of `main.py` to change the input audio, processing duration, SNR, learning rate, filter order, weight initialization, or room filter. If no seed is provided, each run generates different noise.

The processing functions are organized in `processing/`: audio input and output, the ADALINE algorithm, metrics, and plots.

## Output

- `output/images/` contains the PNG plots when plot generation is enabled.
- `output/audios/` contains the noisy signal, recovered signal, and estimated noise as WAV files.

The input SNR describes the noisy signal before filtering. The recovered audio SNR compares the ADALINE output with the original audio; a higher value indicates a closer recovery.