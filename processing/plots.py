import numpy as np
from scipy.ndimage import uniform_filter1d


def save_plots(sample_rate, clean, noisy, recovered, time, weight_history, bias_history, image_dir, show):
    # Load plotting tools only when plots are enabled
    import matplotlib.pyplot as plt

    # Create the image folder and collect figures for saving
    image_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    figure = plt.figure("Original signal")
    plt.plot(time, clean)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title('Original audio signal (source "S")')
    figures.append((figure, "original_signal.png"))

    figure = plt.figure("Noisy signal")
    plt.plot(time, noisy)
    plt.title("Signal with environmental noise (ADALINE target)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    figures.append((figure, "noisy_signal.png"))

    figure = plt.figure("Clean vs recovered")
    plt.plot(time, clean, "k", linewidth=1.5, label="Original clean signal")
    plt.plot(time, recovered, "b", linewidth=1, label='Recovered audio (error "e")')
    plt.legend()
    plt.title("Adaptive noise canceller performance")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    figures.append((figure, "clean_vs_recovered.png"))

    figure = plt.figure("Clean vs noisy vs recovered")
    plt.plot(time, clean, "k", linewidth=1.5, label="Original clean signal")
    plt.plot(time, noisy, "r--", label="Noisy signal")
    plt.plot(time, recovered, "b", linewidth=1, label="Recovered audio")
    plt.legend()
    plt.title("Adaptive noise canceller performance")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    figures.append((figure, "clean_vs_noisy_vs_recovered.png"))

    # Smooth squared error over a 50 ms window for the learning curve
    window = max(1, round(sample_rate * 0.05))
    moving_mse = uniform_filter1d((clean - recovered) ** 2, size=window, mode="nearest")

    figure = plt.figure("Learning curve (dB)")
    plt.plot(time, 10 * np.log10(moving_mse + np.finfo(float).eps), "m", linewidth=1.5)
    plt.title("Learning curve: residual noise attenuation")
    plt.xlabel("Time (s)")
    plt.ylabel("MSE (dB)")
    plt.grid(True)
    figures.append((figure, "learning_curve_db.png"))

    figure = plt.figure("Learning curve")
    plt.plot(time, moving_mse, "m", linewidth=1.5)
    plt.title("Learning curve: error minimization")
    plt.xlabel("Time (s)")
    plt.ylabel("MSE")
    plt.grid(True)
    figures.append((figure, "learning_curve.png"))

    figure = plt.figure("Weights")
    plt.plot(time, weight_history.T, linewidth=1.5)
    plt.title("Time evolution of the synaptic weights")
    plt.xlabel("Time (s)")
    plt.ylabel("Weight amplitude")
    plt.grid(True)
    figures.append((figure, "weights.png"))

    figure = plt.figure("Bias")
    plt.plot(time, bias_history, "k", linewidth=1.5)
    plt.title("Time evolution of the bias")
    plt.xlabel("Time (s)")
    plt.ylabel("Bias amplitude")
    plt.grid(True)
    figures.append((figure, "bias.png"))

    # Save every figure. only open windows when explicitly requested
    for figure, filename in figures:
        figure.savefig(image_dir / filename, dpi=110)

    if show:
        plt.show()
    else:
        for figure, _ in figures:
            plt.close(figure)