import numpy as np
def generate_template(freq, duration=1.0, fs=4096):
    t = np.linspace(0, duration, int(fs * duration))
    template = np.sin(2 * np.pi * freq * t + 0.5 * np.pi * (t**2))
    return template
def matched_filter(data, template):
    correlation = np.correlate(data, template, mode='full')
    snr = correlation / (np.std(data) * np.linalg.norm(template) + 1e-10)
    return snr