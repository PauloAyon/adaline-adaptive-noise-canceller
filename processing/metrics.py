import numpy as np


def calculate_snr(signal, noise):
    # Compare total signal and noise power in decibels
    signal_power = np.sum(np.asarray(signal, dtype=np.float64) ** 2)
    noise_power = np.sum(np.asarray(noise, dtype=np.float64) ** 2)
    if noise_power == 0:
        return float("inf")
    if signal_power == 0:
        return float("-inf")
    return float(10 * np.log10(signal_power / noise_power))


def calculate_mse(reference, estimate):
    # Return the average squared difference between two signals
    difference = np.asarray(reference, dtype=np.float64) - np.asarray(estimate, dtype=np.float64)
    return float(np.mean(difference ** 2))