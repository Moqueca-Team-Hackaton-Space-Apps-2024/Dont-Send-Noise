import numpy as np
import pandas as pd
import scipy 
from typing import Tuple

def FourierTransformation(file_path: str,
                        sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    
    """ Fourier tranformation.

    This function transform the input data through the Fourier transform.

    Args:
        file_path: String containing the path to the csv file where the data from an event is stored.
        sampling_rate: Float with the sampling rate of the measurements (frequency at which the measurements are taken).
        
    Returns:
        Tuple containing:
            - fft_values: np.ndarray containing the complex coefficients of the Fourier Transform.
            - frequencies: np.ndarray containing the discretized frequencies of the Fourier Transform.
    """
    
    data_cat = pd.read_csv(file_path)

    fft_values = np.fft.fft(data_cat['velocity(m/s)'])
    frequencies = np.fft.fftfreq(len(fft_values), d=(1/sampling_rate))
    
    return fft_values, frequencies

def FourierFiltering(fft_values: np.ndarray,
                     frequencies: np.ndarray,
                     N: int = 10,
                     window_size: int = 50) -> np.ndarray:

    """ Filter of the Fourier Transform coefficients.

    This function applies a filter to the Fourier Transform values keeping only the values around the N values with a higher amplitude (module of the complex
    coefficients). The rest of coefficients are set to 0.

    Args:
        fft_values: np.ndarray containing the complex coefficients of the Fourier Transform.
        frequencies: np.ndarray containing the discretized frequencies of the Fourier Transform.
        N: integer indicating the amount of peaks of the amplitude that will be considered by the filter.
        window_size: integer indicating the amount of datapoints that will be considered in each side of each of the peaks.

    Returns:
        filtered_fft: np.ndarray containing the filtered Fourier Transform coefficients.
    """
    
    amplitude = np.abs(fft_values)
    max_amplitude_indices = np.argsort(amplitude)[-N:]
    windows = []
    for i in list(max_amplitude_indices):
        windows.append(list(range(i - window_size, i + window_size)))

    mask = np.zeros(frequencies.shape, dtype=bool)

    for i in windows:
        mask[i] = True
        
    filtered_fft = fft_values * mask

    return filtered_fft

def FourierInverseTransform(filtered_fft: np.ndarray) -> np.ndarray:

    """ Inverse Fourier Transform.

    Function that calculate the Inverse Transform.

    Args:
        filtered_fft: np.ndarray containing the Fourier Transform values that are desired to be transformed back to the original space.

    Returns:
        velocity_transform: np.ndarray containing the transformation of the Fourier Transform values back to the original space.
    """

    velocity_transformed = np.fft.ifft(filtered_fft)

    return velocity_transformed

def PowerSpectralDensityMax(file_path: str,
                            velocity_transformed: np.ndarray,
                            sampling_rate: float,
                            threshold: int = 10000,
                            peak_difference: float = 1.4) -> Tuple[float, bool]:

    """ Power Spectral Density Maximum calculation.

    Function that calculates the Power Spectral Density (PSD) of the data in the original space (time vs. velocity) and calculates the maximum of all its values
    to obtain the time at which this value is highest.

    Args:
        file_path: String containing the path to the csv file where the data from an event is stored.
        velocity_transformed: np.ndarray containing the transformation of the Fourier Transform values back to the original space.
        sampling_rate: Float with the sampling rate of the measurements (frequency at which the measurements are taken).
        threshold: Integer indicating the amount of data points that will be considered on each side of the maximum to check how much higher the module of the
        velocity is in that maximum compared to the average value around that point.
        peak_difference: Float indicating how much higher should the module of the velocity be in the maximum compared to the average of the modules of the 
        points around that maximum in order to consider the point found as a seismic event.

    Returns:
        Tuple containing:
            - start_time: Float indicating the time at which the seismic event starts.
            - seismic_indicator: Boolean value indicating whether the point found corresponds to a seismic event or not.
    """
    
    data_cat = pd.read_csv(file_path)
    seismic_indicator = False
    f, t, sxx = scipy.signal.spectrogram(velocity_transformed, sampling_rate)

    flat_argmax = sxx.argmax()
    indices_2d = np.unravel_index(flat_argmax, sxx.shape)
    start_time = t[indices_2d[1]]
    data_cat_index = np.argmin(np.abs(data_cat['time_rel(sec)'] - t[indices_2d[1]]))
    maximum_velocity = np.abs(velocity_transformed)[data_cat_index]
    velocity_window_mean = np.abs(velocity_transformed)[data_cat_index - threshold : data_cat_index + threshold].mean()

    if (velocity_window_mean.item() * peak_difference) < maximum_velocity.item():
        seismic_indicator = True

    return start_time, seismic_indicator

                     