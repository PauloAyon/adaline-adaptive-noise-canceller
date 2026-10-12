import numpy as np


def run_adaline(reference, desired, filter_order, learning_rate, init_random, rng, store_history):
    # Check that both signals can be processed sample by sample
    reference = np.asarray(reference, dtype=np.float64)
    desired = np.asarray(desired, dtype=np.float64)
    if reference.ndim != 1 or desired.ndim != 1 or reference.size != desired.size:
        raise ValueError("Reference and desired signals must be one-dimensional and equal in size.")
    if reference.size == 0:
        raise ValueError("Input signals must not be empty.")
    if filter_order < 1 or filter_order > reference.size:
        raise ValueError("Filter order must be between 1 and the number of samples.")

    # Start with zero or random weights, as selected
    if init_random:
        weights = rng.random(filter_order)
        bias = float(rng.random())
    else:
        weights = np.zeros(filter_order)
        bias = 0.0

    sample_count = len(reference)
    estimated_noise = np.zeros(sample_count)
    recovered_audio = np.zeros(sample_count)
    # Skip these large arrays when no plots are requested
    weight_history = np.zeros((filter_order, sample_count)) if store_history else None
    bias_history = np.zeros(sample_count) if store_history else None

    for sample in range(filter_order - 1, sample_count):
        # Build the current input vector from the newest sample and its delays
        input_vector = reference[sample - filter_order + 1 : sample + 1][::-1]
        estimated_noise[sample] = np.dot(weights, input_vector) + bias

        # Subtract estimated noise; the residual is the recovered audio
        recovered_audio[sample] = desired[sample] - estimated_noise[sample]

        # Update weights and bias with the LMS learning rule
        weights = weights + learning_rate * recovered_audio[sample] * input_vector
        bias = bias + learning_rate * recovered_audio[sample]

        if store_history:
            weight_history[:, sample] = weights
            bias_history[sample] = bias

    # Return signals, final parameters, and optional plotting history
    return estimated_noise, recovered_audio, weights, bias, weight_history, bias_history