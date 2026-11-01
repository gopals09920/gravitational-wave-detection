from src.preprocessing import butter_bandpass_filter
from src.matched_filter import generate_template, matched_filter
from src.model import GravitationalWaveAutoencoder
import torch 
import numpy as np
fs = 4096
t = np.linspace(0, 1, fs)
dummy_data = np.sin(2 * np.pi * 50 * t) + np.random.normal(0, 0.1, fs)
filtered_data = butter_bandpass_filter(dummy_data, lowcut = 20.0, highcut = 300.0, fs = fs)
model = GravitationalWaveAutoencoder()
tensor_data = torch.tensor(filtered_data.copy(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
reconstructed = model(tensor_data)
print("Deep Learning Model successfully tested! Output tensor shape:", reconstructed.shape)

template = generate_template(freq = 50, duration = 1.0, fs = fs)
snr_output = matched_filter(filtered_data, template)
print("Matched Filter successfully executed! Max SNR peak:", np.max(snr_output))