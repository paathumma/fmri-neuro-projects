import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#1. simulated neural data

# Define time parameters and basic components
duration = 4.0
fs = 250.0
n_samples = int(fs*duration)
time = np.linspace(0, duration, n_samples, endpoint=False)

# Component A: low-frequency baseline drift (0.2 Hz slow wwave)
baseline_drift = 3.0 * np.sin(2 * np.pi * 0.2 *time)

#Componet B: target neural rhythm
alpha_rhythm = 1.5 * np.sin(2 * np.pi * 10.0 * time)

# Component C: Power linenoise (50 Hz electrical grid interference)
power_line = 0.8 * np.sin(2 * np.pi * 50.0 *time)

# Component D: Background noise
background_noise = np.random.normal(loc=0.0, scale=0.6, size=n_samples)

# Component E: Transient spike events (e.g., ERP components or spikes)
spikes = np.zeros(n_samples)
spike_indices = [200, 500, 800]
spikes[spike_indices] = [4.0, -3.5, 5.0]

# Combine into a single "Raw" neural signal
raw_neural_signal = baseline_drift + alpha_rhythm + power_line + background_noise + spikes


# 2. NEURAL FILTERING PIPELINE (USING FILTFILT)

# Step 1: High-Pass Filter (Remove slow baseline drift < 0.5 Hz)
b_hp, a_hp = signal.butter(4, 0.5, btype='high', fs=fs)
filtered_hp = signal.filtfilt(b_hp, a_hp, raw_neural_signal)

# Step 2: Notch Filter (Remove 50 Hz electrical line noise)
# Q-factor controls how narrow/sharp the notch filter is
b_notch, a_notch = signal.iirnotch(w0=50.0, Q=30.0, fs=fs)
filtered_notch = signal.filtfilt(b_notch, a_notch, filtered_hp)

# Step 3: Bandpass Filter (Isolate Alpha Band: 8.0 Hz to 13.0 Hz)
b_bp, a_bp = signal.butter(4, [8.0, 13.0], btype='band', fs=fs)
alpha_band_signal = signal.filtfilt(b_bp, a_bp, filtered_notch)



# 3. VISUALIZATION

plt.figure(figsize=(12, 8))

# Plot 1: Raw vs Cleaned Comparison
plt.subplot(2, 1, 1)
plt.plot(time, raw_neural_signal, label="Raw Neural Signal (Noisy)", color="purple", alpha=0.5)
plt.plot(time, filtered_notch, label="Cleaned Signal (Drift & 50Hz Removed)", color="blue", linewidth=1.5)
plt.title("Neural Signal Processing: Artifact Removal", fontsize=13, fontweight='bold')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (uV)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper right")

# Plot 2: Isolated Neural Rhythm (Alpha Band)
plt.subplot(2, 1, 2)
plt.plot(time, alpha_band_signal, label="Isolated Alpha Rhythm (8-13 Hz)", color="green", linewidth=1.5)
plt.title("Zero-Phase Bandpass Filtered Output", fontsize=13, fontweight='bold')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (uV)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper right")

plt.tight_layout()
plt.show()



