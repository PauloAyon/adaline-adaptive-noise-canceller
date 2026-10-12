import argparse
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

from processing.adaline import run_adaline
from processing.audio import read_audio, save_audio
from processing.metrics import calculate_mse, calculate_snr


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
AUDIO_FILE = PROJECT_DIR / "blackMagicWoman_WAV.wav"
IMAGE_DIR = PROJECT_DIR / "output" / "images"
AUDIO_OUTPUT_DIR = PROJECT_DIR / "output" / "audios"

# Parameters
SECONDS_TO_PROCESS = 20
SNR_DB = 3
LEARNING_RATE = 0.002
FILTER_ORDER = 10
INIT_RANDOM = False
ROOM_FILTER = (0.8, -0.5, 0.2)
RANDOM_SEED = None


def parse_arguments():
    # Read optional run modes from the command line
    parser = argparse.ArgumentParser(description="Adaptive noise cancellation with ADALINE.")
    plot_options = parser.add_mutually_exclusive_group()
    plot_options.add_argument("--no-plots", action="store_true", help="Skip creating and saving plots.")
    plot_options.add_argument("--show-plots", action="store_true", help="Show plots after saving them.")
    parser.add_argument("--play", action="store_true", help="Play the generated audio files.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Set a random seed for repeatable results.")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Load and validate the source audio
    try:
        sample_rate, clean = read_audio(AUDIO_FILE, SECONDS_TO_PROCESS)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Error loading input audio: {error}") from error

    signal_power = np.mean(clean ** 2)
    if signal_power == 0:
        raise SystemExit("Error: the input audio has no measurable signal power.")

    # Create reference noise at the requested signal-to-noise ratio
    rng = np.random.default_rng(args.seed)
    snr_linear = 10 ** (SNR_DB / 10)
    noise_standard_deviation = np.sqrt(signal_power / snr_linear)
    reference_noise = rng.normal(0, noise_standard_deviation, len(clean))
    filtered_noise = lfilter(ROOM_FILTER, 1, reference_noise)
    noisy = clean + filtered_noise

    # Keep weight history only when plots need it
    make_plots = not args.no_plots
    time = np.arange(len(clean)) / sample_rate
    (
        estimated_noise,
        recovered,
        weights,
        bias,
        weight_history,
        bias_history,
    ) = run_adaline(
        reference_noise,
        noisy,
        FILTER_ORDER,
        LEARNING_RATE,
        INIT_RANDOM,
        rng,
        make_plots,
    )

    # Measure the noisy input and recovered output
    input_snr = calculate_snr(clean, noisy - clean)
    output_snr = calculate_snr(clean, recovered - clean)
    mse = calculate_mse(clean, recovered)

    print(f"Filter order = {FILTER_ORDER}, learning rate = {LEARNING_RATE}, SNR = {SNR_DB} dB")
    print(f"Input SNR: {input_snr:.2f} dB")
    print(f"Recovered audio SNR: {output_snr:.2f} dB")
    print(f"Recovered audio MSE: {mse:.6f}")
    print("Final weights:", np.round(weights, 3))
    print("Room filter:", ROOM_FILTER)

    # Save audio files in the output audio folder
    save_audio(AUDIO_OUTPUT_DIR / "noisy.wav", sample_rate, noisy)
    save_audio(AUDIO_OUTPUT_DIR / "recovered.wav", sample_rate, recovered)
    save_audio(AUDIO_OUTPUT_DIR / "estimated_noise.wav", sample_rate, estimated_noise)

    if make_plots:
        from processing.plots import save_plots

        # Save plots by default and display them only when requested
        save_plots(
            sample_rate,
            clean,
            noisy,
            recovered,
            time,
            weight_history,
            bias_history,
            IMAGE_DIR,
            args.show_plots,
        )

    if args.play:
        # Load the optional playback library only when needed
        try:
            import sounddevice as sd
        except ImportError as error:
            raise SystemExit("Audio playback requires sounddevice. Install requirements-audio.txt.") from error

        for label, signal in (
            ("Noisy signal", noisy),
            ("Recovered signal", recovered),
            ("Estimated noise", estimated_noise),
        ):
            print(f"Playing {label.lower()}...")
            peak = np.max(np.abs(signal))
            sd.play(signal / peak if peak else signal, sample_rate)
            sd.wait()


if __name__ == "__main__":
    main()