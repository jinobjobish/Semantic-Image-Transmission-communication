# utils/channel_simulation.py
import numpy as np

class AWGNChannel:
    """Additive White Gaussian Noise Channel"""
    def __init__(self, snr_db=10.0):
        self.snr_db = snr_db
    
    def transmit(self, data):
        """Add AWGN noise to transmitted data"""
        elements, indices = data
        
        if len(elements) == 0:
            return data
        
        # Calculate noise power based on SNR
        signal_power = np.mean(elements**2)
        snr_linear = 10**(self.snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Add Gaussian noise
        noise = np.random.normal(0, np.sqrt(noise_power), elements.shape)
        noisy_elements = elements + noise
        
        return noisy_elements, indices

class RayleighChannel:
    """Rayleigh Fading Channel"""
    def __init__(self, snr_db=10.0):
        self.snr_db = snr_db
    
    def transmit(self, data):
        """Apply Rayleigh fading"""
        elements, indices = data
        
        if len(elements) == 0:
            return data
        
        # Rayleigh fading coefficient (complex)
        h = (np.random.randn(len(elements)) + 1j * np.random.randn(len(elements))) / np.sqrt(2)
        
        # Calculate noise power
        signal_power = np.mean(np.abs(elements)**2)
        snr_linear = 10**(self.snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Apply fading and noise
        faded_signal = elements * np.abs(h)
        noise = np.random.normal(0, np.sqrt(noise_power), elements.shape)
        received_signal = faded_signal + noise
        
        return received_signal, indices