import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.io import wavfile

def relativistic_signal_processor(input_audio, output_distorted, output_restored):
    # Parameters
    c = 3e8  # Speed of light
    a = 3e7  # Proper acceleration (Set aggressively high to hear the effect quickly)

    # Phase 1: Signal Generation
    sample_rate, signal_src = wavfile.read(input_audio)
    
    # Convert to mono if the audio is stereo
    if len(signal_src.shape) > 1:
        signal_src = signal_src.mean(axis=1)

    N_src = len(signal_src)
    T_total = N_src / sample_rate
    t_emit = np.linspace(0, T_total, N_src)

    # Phase 2: The Channel Model (Forward Problem)
    print("Simulating relativistic channel distortion...")
    # 1. Trajectory and arrival times
    distance = (c**2 / a) * np.sqrt(1 + (a * t_emit / c)**2)
    t_rx = t_emit + distance / c

    # 2. Create the Receiver's Uniform Grid
    # We scale the number of points by the time dilation factor to ensure 
    # the actual audio duration stretches out when played at the same sample rate.
    T_rx_total = max(t_rx) - min(t_rx)
    N_rx = int(T_rx_total * sample_rate)
    t_rx_uniform = np.linspace(min(t_rx), max(t_rx), N_rx)

    # 3. Interpolate (What the receiver hears)
    interpolator_fwd = interp1d(t_rx, signal_src, kind='cubic', fill_value="extrapolate")
    signal_distorted = interpolator_fwd(t_rx_uniform)

    # Normalize and export distorted audio
    sig_dist_norm = np.int16((signal_distorted / np.max(np.abs(signal_distorted))) * 32767)
    wavfile.write(output_distorted, sample_rate, sig_dist_norm)

    # Phase 3: The Restoration (Inverse Problem)
    print("Applying matched filter to restore signal...")
    # 1. Invert the equation tr(te) to find te(tr)
    # Using the analytically derived inverse formula
    t_emit_recovered = (t_rx_uniform**2 - (c/a)**2) / (2 * t_rx_uniform)

    # 2. Resample distorted signal back onto the te domain
    interpolator_inv = interp1d(t_emit_recovered, signal_distorted, kind='cubic', fill_value="extrapolate")
    signal_restored = interpolator_inv(t_emit)

    # Normalize and export restored audio
    sig_rest_norm = np.int16((signal_restored / np.max(np.abs(signal_restored))) * 32767)
    wavfile.write(output_restored, sample_rate, sig_rest_norm)

        # Phase 4: Spectrogram Deliverables
    print("Generating spectrograms...")
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Define a common max time across all signals
    T_common = max(T_total, T_rx_total)

    # Original
    axs[0].specgram(signal_src, Fs=sample_rate, NFFT=1024, cmap='inferno')
    axs[0].set_title("Original Message ($t_{emit}$)")
    axs[0].set_ylabel("Frequency (Hz)")
    axs[0].set_xlim(0, T_common)
    axs[0].tick_params(labelbottom=True)  # <-- force x labels

    # Distorted
    axs[1].specgram(signal_distorted, Fs=sample_rate, NFFT=1024, cmap='inferno')
    axs[1].set_title("Received Message (Distorted at $t_{rx}$)")
    axs[1].set_ylabel("Frequency (Hz)")
    axs[1].set_xlim(0, T_common)
    axs[1].tick_params(labelbottom=True)  # <-- force x labels

    # Restored
    axs[2].specgram(signal_restored, Fs=sample_rate, NFFT=1024, cmap='inferno')
    axs[2].set_title("Restored Message (Inverted to $t_{emit}$)")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Frequency (Hz)")
    axs[2].set_xlim(0, T_common)

    plt.tight_layout()
    plt.savefig("spectrograms.png", dpi=300, bbox_inches='tight')

# Run the processor with the specified audio files
relativistic_signal_processor('houston.wav', 'demonic_rx.wav', 'restored.wav')