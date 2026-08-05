import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 1. Define time parameters and basic components
n_samples = 1000
time = np.linspace(0, 10, n_samples, endpoint=False)

# 2. Create the 5 Hz square wave
# (Using 'time' instead of 't' so array dimensions match later)
square_wave = signal.square(2 * np.pi * 5 * time)

# 3. Create the spike signal
spike_signal = np.zeros(n_samples)
spike_indices = [150, 400, 750, 900]
spike_amplitudes = [5.0, -3.0, 7.5, 4.0]
spike_signal[spike_indices] = spike_amplitudes

# 4. Generate white noise
white_noise = np.random.normal(loc=0.0, scale=1.0, size=n_samples)

# 5. Combine all signals
sign_data = white_noise + square_wave + spike_signal

# 6. Plot and visualize the combined data
plt.figure(figsize=(10, 5))
plt.plot(time, sign_data, label="Combined Signals", color="purple")

plt.title("Visualisation of Combined Peak and White Noise Signal", fontsize=14, fontweight='bold')
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=11, loc="upper right")
plt.tight_layout()

# This will now display properly without freezing your terminal
plt.show()


import scipy.signal as signal
import matplotlib.pyplot as plt

# 1. Design the High-Pass Filter 
# (Using 50 Hz as a cutoff example; adjust as needed for your data)
b_high, a_high = signal.butter(4, 50, btype='high', fs=1000)

# 2. Apply the filter to sign_data
highw = signal.lfilter(b_high, a_high, sign_data)

# 3. Create a brand-new, clean plot window
plt.figure()

# 4. Plot both signals using your 'time' variable
plt.plot(time, sign_data, label="Combined Signals", color="purple", alpha=0.5)
plt.plot(time, highw, label="High-Pass Filtered Signal", color="green")

# 5. Format and display
plt.title("IIR High-Pass Filtering")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True, linestyle='--')
plt.legend()
plt.show()

# 1. Design
b_low, a_low = signal.butter(4, 100, btype='low', fs=1000)
# 2. Implement with filtfilt (zero phase delay)
loww = signal.filtfilt(b_low, a_low, sign_data)
