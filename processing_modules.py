import numpy as np
from scipy import signal
from scipy.io import wavfile
import librosa
import soundfile as sf
import tempfile
import os
import midiutil
from typing import Tuple, Optional, Dict, Any

class PianoEffect:
    def __init__(self, 
                 hammer_hardness: float = 0.7,
                 string_resonance: float = 0.8,
                 brightness: float = 0.6,
                 sustain_level: float = 0.7,
                 room_size: float = 0.5,
                 damper_position: float = 0.8,
                 compression_ratio: float = 4.0,
                 compression_attack: float = 0.01,
                 compression_release: float = 0.2):
        """
        Initialize the piano effect processor.
        
        Args:
            hammer_hardness: Controls the attack characteristic (0-1)
            string_resonance: Controls the resonance of the strings (0-1)
            brightness: Controls the high frequency content (0-1)
            sustain_level: Controls the decay time (0-1)
            room_size: Controls the reverb size (0-1)
            damper_position: Controls the damping characteristics (0-1)
            compression_ratio: Compression ratio for dynamic control
            compression_attack: Compressor attack time in seconds
            compression_release: Compressor release time in seconds
        """
        self.hammer_hardness = hammer_hardness
        self.string_resonance = string_resonance
        self.brightness = brightness
        self.sustain_level = sustain_level
        self.room_size = room_size
        self.damper_position = damper_position
        self.compression_ratio = compression_ratio
        self.compression_attack = compression_attack
        self.compression_release = compression_release

    def apply_hammer_effect(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply hammer strike characteristics."""
        # Calculate attack envelope based on hammer hardness
        attack_time = 0.005 * (1 - self.hammer_hardness)  # Harder hammer = faster attack
        attack_samples = int(attack_time * sr)
        
        # Create attack envelope
        attack_env = np.linspace(0, 1, attack_samples)
        attack_env = np.power(attack_env, 2 * self.hammer_hardness)  # Shape the attack curve
        
        # Apply to audio
        if len(audio) > attack_samples:
            audio[:attack_samples] *= attack_env
        return audio

    def apply_string_resonance(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply string resonance filtering."""
        # Create resonant filter
        resonance_freq = 1000 + 2000 * self.string_resonance
        q_factor = 1 + 9 * self.string_resonance  # Higher resonance = sharper peak
        
        b, a = signal.iirpeak(resonance_freq, q_factor, sr)
        return signal.lfilter(b, a, audio)

    def apply_brightness(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply brightness control using a shelf filter."""
        # High shelf filter parameters
        cutoff_freq = 2000  # Hz
        gain_db = 12 * (self.brightness - 0.5)  # Convert to dB gain
        
        b, a = signal.iirfilter(2, cutoff_freq, btype='highshelf', ftype='butter', fs=sr, output='ba', gain=gain_db)
        return signal.lfilter(b, a, audio)

    def apply_sustain(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply sustain envelope."""
        # Calculate decay time based on sustain level
        decay_time = 0.5 + 2.5 * self.sustain_level  # 0.5 to 3 seconds decay
        decay_samples = int(decay_time * sr)
        
        # Create decay envelope
        decay_env = np.exp(-3 * np.linspace(0, 1, decay_samples))
        decay_env = np.power(decay_env, 1 - self.sustain_level)  # Adjust decay curve
        
        # Apply to audio
        if len(audio) > decay_samples:
            audio[decay_samples:] *= decay_env[-1]  # Sustain level for the rest
            audio[:decay_samples] *= decay_env
        return audio

    def apply_room(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply room simulation using basic reverb."""
        # Room size affects delay times and decay
        room_size_ms = 20 + 80 * self.room_size  # 20-100ms
        num_reflections = int(3 + 5 * self.room_size)  # 3-8 reflections
        
        # Create multiple delayed copies with decreasing amplitude
        result = audio.copy()
        for i in range(num_reflections):
            delay_samples = int((room_size_ms * (i + 1) / 1000) * sr)
            amplitude = 0.3 * (1 - self.room_size) ** i
            delayed = np.pad(audio, (delay_samples, 0))[:len(audio)] * amplitude
            result += delayed
            
        return result / (1 + num_reflections * 0.3)  # Normalize

    def apply_damper(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply damper position effect using filtering."""
        # Damper position affects high frequency content
        cutoff_freq = 500 + 7500 * self.damper_position  # Higher position = more high frequencies
        order = 4
        
        b, a = signal.butter(order, cutoff_freq, btype='lowpass', fs=sr)
        return signal.lfilter(b, a, audio)

    def apply_compression(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply dynamic range compression."""
        # Parameters
        threshold = 0.3
        ratio = self.compression_ratio
        attack_samples = int(self.compression_attack * sr)
        release_samples = int(self.compression_release * sr)
        
        # Envelope follower
        abs_audio = np.abs(audio)
        env = np.zeros_like(audio)
        for i in range(1, len(audio)):
            if abs_audio[i] > env[i-1]:
                env[i] = env[i-1] + (abs_audio[i] - env[i-1]) / attack_samples
            else:
                env[i] = env[i-1] + (abs_audio[i] - env[i-1]) / release_samples
        
        # Compression curve
        gain = np.ones_like(audio)
        mask = env > threshold
        gain[mask] = threshold + (env[mask] - threshold) / ratio
        gain[mask] /= env[mask]
        
        return audio * gain

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Process the audio through the piano effect chain."""
        # Normalize input
        audio = audio / np.max(np.abs(audio))
        
        # Apply effects chain
        audio = self.apply_hammer_effect(audio, sr)
        audio = self.apply_string_resonance(audio, sr)
        audio = self.apply_brightness(audio, sr)
        audio = self.apply_sustain(audio, sr)
        audio = self.apply_damper(audio, sr)
        audio = self.apply_room(audio, sr)
        audio = self.apply_compression(audio, sr)
        
        # Final normalization
        return audio / np.max(np.abs(audio))

class TalkboxEffect:
    # ... existing code ...

    def process_audio(self, audio_path: str, output_path: str, params: Dict[str, Any]) -> None:
        """Process the audio file with the talkbox effect."""
        # Load the audio file
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Check if we should use piano mode
        if params.get('instrument_type', 'guitar') == 'piano':
            # Initialize and apply piano effect
            piano = PianoEffect(
                hammer_hardness=params.get('hammer_hardness', 0.7),
                string_resonance=params.get('string_resonance', 0.8),
                brightness=params.get('brightness', 0.6),
                sustain_level=params.get('sustain_level', 0.7),
                room_size=params.get('room_size', 0.5),
                damper_position=params.get('damper_position', 0.8),
                compression_ratio=params.get('compression_ratio', 4.0),
                compression_attack=params.get('compression_attack', 0.01),
                compression_release=params.get('compression_release', 0.2)
            )
            processed_audio = piano.process(audio, sr)
        else:
            # Apply guitar talkbox effect
            processed_audio = self.process(
                audio,
                sr,
                drive=params.get('drive', 2.5),
                wah_min_freq=params.get('wah_min_freq', 400),
                wah_max_freq=params.get('wah_max_freq', 2000),
                wah_speed=params.get('wah_speed', 3.0),
                wah_depth=params.get('wah_depth', 0.8),
                wah_resonance=params.get('wah_resonance', 8.0),
                pre_emphasis_freq=params.get('pre_emphasis_freq', 1200),
                compression_ratio=params.get('compression_ratio', 6.0),
                compression_attack=params.get('compression_attack', 0.005),
                compression_release=params.get('compression_release', 0.15)
            )
        
        # Save the processed audio
        sf.write(output_path, processed_audio, sr)

# ... rest of the existing code ... 