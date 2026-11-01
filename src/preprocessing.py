import os
import numpy as np
from scipy.signal import butter, filtfilt
def load_ligo_data(file_path):
    import h5py
    with h5py.File(file_path, 'r') as f:
        strain = f['strain']['Strain'][...]
        gps_start = f['meta']['GPSstart'][()]
        duration = f['meta']['Duration'][()]
        sample_rate = f['meta']['GPSduration'][()]
    return strain, gps_start, duration
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype = 'band')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = filtfilt(b, a, data)
    return filtered_data
def compute_whitening(strain, fs):
    fft_vals = np.fft.rfft(strain)
    freqs = np.fft.rfftfreq(len(strain), 1/fs)
    return freqs, fft_vals