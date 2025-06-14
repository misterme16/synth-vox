from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import numpy as np
import soundfile as sf
import librosa
import pyworld as world
from scipy import signal
import io
import base64
from typing import Dict, Optional, Union
import mido
from mido import MidiFile, MidiTrack, Message, bpm2tempo
import tempfile
import os

app = FastAPI()

class AudioRequest(BaseModel):
    audio: str
    type: str
    params: Dict[str, float] = {}

class AudioResponse(BaseModel):
    audio: str
    sampleRate: int

class ErrorResponse(BaseModel):
    error: str

def ensure_mono(audio_data):
    """Convert stereo audio to mono if necessary."""
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        return np.mean(audio_data, axis=1)
    return audio_data

def convert_to_midi(audio_data, sample_rate, bpm=120):
    """
    Convert audio data to MIDI notes.
    Returns MIDI file as bytes.
    """
    print("Starting MIDI conversion process...")
    
    try:
        # Ensure mono audio
        audio_data = ensure_mono(audio_data)
        
        # Perform pitch detection
        print("Detecting pitch...")
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_data,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C6'),
            sr=sample_rate,
            frame_length=2048,
            win_length=1024,
            hop_length=512
        )
        times = librosa.times_like(f0, sr=sample_rate, hop_length=512)
        f0 = np.nan_to_num(f0)
        print("Pitch detection completed")
        
        # Quantize pitches to reduce note density
        print("Quantizing notes...")
        midi_notes = []
        current_note = None
        note_start_time = 0
        min_note_duration = 0.1  # Minimum note duration in seconds
        
        for t, pitch, voiced in zip(times, f0, voiced_flag):
            if pitch > 0 and voiced:
                midi_note = int(round(librosa.hz_to_midi(pitch)))
                if current_note != midi_note:
                    if current_note is not None:
                        # Only add notes that are long enough
                        if t - note_start_time >= min_note_duration:
                            midi_notes.append((note_start_time, t, current_note))
                    current_note = midi_note
                    note_start_time = t
            elif current_note is not None:
                if t - note_start_time >= min_note_duration:
                    midi_notes.append((note_start_time, t, current_note))
                current_note = None
        
        # Create MIDI file with quantized notes
        print("Creating MIDI file...")
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage('set_tempo', tempo=bpm2tempo(bpm)))
        
        ticks_per_second = mid.ticks_per_beat * (bpm / 60.0)
        
        for start_time, end_time, note in midi_notes:
            # Convert time to ticks
            start_ticks = int(start_time * ticks_per_second)
            end_ticks = int(end_time * ticks_per_second)
            duration = end_ticks - start_ticks
            
            if duration > 0:
                track.append(Message('note_on', note=note, velocity=100, time=0))
                track.append(Message('note_off', note=note, velocity=64, time=duration))
        
        if not midi_notes:
            raise Exception("No valid notes detected in the audio")
        
        # Save MIDI to bytes
        with io.BytesIO() as midi_buffer:
            mid.save(file=midi_buffer)
            return midi_buffer.getvalue()
            
    except Exception as e:
        raise Exception(f"Failed during MIDI conversion: {str(e)}")

def create_gaussian_window(M, std):
    """Create a Gaussian window of length M with given standard deviation."""
    n = np.arange(0, M) - (M-1)/2
    return np.exp(-0.5 * (n/std)**2)

def tube_saturate(x, drive=2.0):
    """
    Simulate tube-style saturation for warmer sound.
    """
    return np.tanh(x * drive) / np.tanh(drive)

def tube_distortion(x, drive=2.0):
    """Simulate guitar amp distortion"""
    # Add even harmonics (like a tube amp)
    drive = np.clip(drive, 0.1, 10.0)
    x_clip = np.clip(x * drive, -1, 1)
    return np.tanh(x_clip) + 0.1 * np.sign(x_clip) * x_clip**2

def create_wah_filter(freq, Q, sample_rate):
    """Create a wah-wah filter at specified frequency"""
    nyquist = sample_rate / 2
    freq = np.clip(freq, 20, nyquist-20)
    b, a = signal.iirpeak(freq, Q, sample_rate)
    return b, a

def bandpass_filter(signal_data, low, high, sr, order=4):
    """Enhanced bandpass filter with steeper rolloff"""
    nyquist = 0.5 * sr
    low_normalized = low / nyquist
    high_normalized = high / nyquist
    
    # Ensure frequencies are within valid range
    low_normalized = np.clip(low_normalized, 0.001, 0.99)
    high_normalized = np.clip(high_normalized, 0.001, 0.99)
    
    # Ensure low is less than high
    if low_normalized >= high_normalized:
        low_normalized, high_normalized = 0.001, 0.99
        
    b, a = signal.butter(order, [low_normalized, high_normalized], btype='band')
    return signal.filtfilt(b, a, signal_data)

class TalkboxEffect:
    def __init__(self, drive=2.5, wah_min_freq=400, wah_max_freq=2000, wah_speed=3.0,
                 wah_depth=0.8, wah_resonance=8.0, pre_emphasis_freq=1200,
                 compression_ratio=6.0, compression_attack=0.005, compression_release=0.15):
        self.drive = drive
        self.wah_min_freq = wah_min_freq
        self.wah_max_freq = wah_max_freq
        self.wah_speed = wah_speed
        self.wah_depth = wah_depth
        self.wah_resonance = wah_resonance
        self.pre_emphasis_freq = pre_emphasis_freq
        self.compression_ratio = compression_ratio
        self.compression_attack = compression_attack
        self.compression_release = compression_release

    def compress_signal(self, x, threshold, ratio, attack_time, release_time, sample_rate):
        """Enhanced compression with smoother envelope"""
        attack_samples = int(attack_time * sample_rate)
        release_samples = int(release_time * sample_rate)
        
        x_abs = np.abs(x)
        mask = x_abs > threshold
        gain_reduction = np.zeros_like(x)
        gain_reduction[mask] = (x_abs[mask] - threshold) * (1 - 1/ratio)
        
        window = np.exp(-np.arange(release_samples) / (release_samples/8))
        window = window / window.sum()
        smoothed_gain = signal.filtfilt(window, [1.0], gain_reduction)
        
        output = x.copy()
        output[mask] = x[mask] - np.sign(x[mask]) * smoothed_gain[mask]
        return output

    def process_audio(self, audio_data, sample_rate):
        """Enhanced guitar-style talkbox implementation"""
        try:
            # Convert to mono if stereo
            audio_data = ensure_mono(audio_data)
            
            # Convert to float32 if needed
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize input
            max_val = np.abs(audio_data).max()
            if max_val > 1.0:
                audio_data = audio_data / max_val
            
            # Create filter chain
            nyquist = sample_rate / 2
            
            # Pre-emphasis filter (guitar pickup simulation)
            b_pre, a_pre = signal.butter(1, self.pre_emphasis_freq/nyquist, btype='high')
            
            # Initial bandpass to focus on guitar frequencies
            b_bandpass, a_bandpass = signal.butter(4, [80/nyquist, 6000/nyquist], btype='band')
            
            # Apply pre-emphasis
            audio_pre = signal.filtfilt(b_pre, a_pre, audio_data)
            
            # Apply initial bandpass
            filtered = signal.filtfilt(b_bandpass, a_bandpass, audio_pre)
            
            # Create time-varying wah filter
            t = np.arange(len(filtered)) / sample_rate
            
            # Generate wah frequency modulation (enhanced)
            wah_freq = self.wah_min_freq + (self.wah_max_freq - self.wah_min_freq) * \
                      (0.5 + 0.5 * np.sin(2 * np.pi * self.wah_speed * t))
            
            # Apply time-varying wah filter
            wahhed = filtered.copy()
            chunk_size = 2048  # Process in chunks for better control
            
            for i in range(0, len(filtered), chunk_size):
                chunk = filtered[i:i + chunk_size]
                chunk_freq = wah_freq[i:i + chunk_size].mean()
                b_wah, a_wah = create_wah_filter(chunk_freq, self.wah_resonance, sample_rate)
                wahhed[i:i + chunk_size] = signal.filtfilt(b_wah, a_wah, chunk) * self.wah_depth + \
                                         chunk * (1 - self.wah_depth)
            
            # Add slight pitch modulation (vibrato)
            vibrato_rate = 6.0
            vibrato_depth = 0.001
            vibrato = 1.0 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
            modulated = wahhed * vibrato
            
            # Apply compression
            compressed = self.compress_signal(modulated, 0.1, self.compression_ratio,
                                           self.compression_attack, self.compression_release, sample_rate)
            
            # Apply tube-style distortion
            distorted = tube_distortion(compressed, drive=self.drive)
            
            # Final cleanup filter
            b_final, a_final = signal.butter(2, [60/nyquist, 7000/nyquist], btype='band')
            final = signal.filtfilt(b_final, a_final, distorted)
            
            # Normalize output
            final = final / np.max(np.abs(final)) * 0.95
            
            return final.astype(np.float32)
            
        except Exception as e:
            print(f"Error during talkbox processing: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Talkbox processing failed: {str(e)}")

class Vocoder:
    def __init__(self, formant_shift=1.0, spectral_envelope=1.0, frequency_range=1.0,
                 aperiodicity=0.5, clarity=0.7):
        self.formant_shift = max(0.5, min(2.0, formant_shift))
        self.spectral_envelope = max(0.5, min(2.0, spectral_envelope))
        self.frequency_range = max(0.5, min(2.0, frequency_range))
        self.aperiodicity = max(0.1, min(1.0, aperiodicity))
        self.clarity = max(0.1, min(1.0, clarity))

    def process_audio(self, audio_data, sample_rate):
        # Convert to mono if stereo
        audio_data = ensure_mono(audio_data)
        
        # Convert to float64 for WORLD
        audio_data = audio_data.astype(np.float64)
        
        # Normalize input
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        try:
            # Extract features
            f0, sp, ap = world.wav2world(audio_data, sample_rate)
            
            # Modify features
            f0_mod = f0 * self.frequency_range
            sp_mod = np.zeros_like(sp)
            
            # Apply formant shift and spectral envelope modification
            for t in range(sp.shape[0]):
                freq_axis = np.linspace(0, 1, sp.shape[1])
                shifted_freq = freq_axis * self.formant_shift
                sp_mod[t] = np.interp(freq_axis, shifted_freq, sp[t]) * self.spectral_envelope
            
            # Modify aperiodicity
            ap_mod = ap * self.aperiodicity
            
            # Enhance clarity
            sp_mod = np.power(sp_mod, self.clarity)
            
            # Synthesize
            audio_data = world.synthesize(f0_mod, sp_mod, ap_mod, sample_rate)
            
            # Normalize output
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val * 0.95
            
            return audio_data.astype(np.float32)
            
        except Exception as e:
            print(f"Error in vocoder processing: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Vocoder processing failed: {str(e)}")

@app.post("/api/process", response_model=Union[AudioResponse, ErrorResponse])
async def process_audio_endpoint(request: AudioRequest):
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio)
        audio_io = io.BytesIO(audio_bytes)
        audio_data, sample_rate = sf.read(audio_io)
        
        # Process based on type
        if request.type == 'talkbox':
            processor = TalkboxEffect(**request.params)
            processed_audio = processor.process_audio(audio_data, sample_rate)
            
            # Convert processed audio to WAV bytes
            output_io = io.BytesIO()
            sf.write(output_io, processed_audio, sample_rate, format='WAV')
            output_bytes = output_io.getvalue()
            
        elif request.type == 'vocoder':
            processor = Vocoder(**request.params)
            processed_audio = processor.process_audio(audio_data, sample_rate)
            
            # Convert processed audio to WAV bytes
            output_io = io.BytesIO()
            sf.write(output_io, processed_audio, sample_rate, format='WAV')
            output_bytes = output_io.getvalue()
            
        elif request.type == 'midi':
            # Get BPM from params or use default
            bpm = request.params.get('bpm', 120)
            
            # Convert to MIDI
            output_bytes = convert_to_midi(audio_data, sample_rate, bpm=bpm)
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown process type: {request.type}")
        
        # Convert to base64
        output_base64 = base64.b64encode(output_bytes).decode('utf-8')
        
        return AudioResponse(audio=output_base64, sampleRate=sample_rate)
        
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 