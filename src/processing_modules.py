import librosa
import numpy as np
import soundfile as sf
import mido
from mido import MidiFile, MidiTrack, Message, bpm2tempo
import os
from scipy.signal import butter, lfilter, hilbert
import time
import wave
from pydub import AudioSegment
import scipy.signal
import pyworld as pw
from scipy import signal

def load_audio(audio_path):
    """Robust audio loading function that tries multiple methods"""
    print(f"\nAttempting to load audio file: {audio_path}")
    print(f"File exists: {os.path.exists(audio_path)}")
    print(f"File size: {os.path.getsize(audio_path) if os.path.exists(audio_path) else 'N/A'} bytes")
    
    try:
        # First try soundfile
        try:
            print("\nTrying SoundFile...")
            y, sr = sf.read(audio_path)
            if len(y.shape) > 1:  # If stereo, convert to mono
                y = y.mean(axis=1)
            print(f"SoundFile success! Sample rate: {sr}Hz, Length: {len(y)} samples")
            return y, sr
        except Exception as e:
            print(f"SoundFile loading failed: {str(e)}")

        # Try librosa as fallback
        try:
            print("\nTrying Librosa...")
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            print(f"Librosa success! Sample rate: {sr}Hz, Length: {len(y)} samples")
            return y, sr
        except Exception as e:
            print(f"Librosa loading failed: {str(e)}")

        # Try pydub
        try:
            print("\nTrying Pydub...")
            audio = AudioSegment.from_file(audio_path)
            samples = np.array(audio.get_array_of_samples())
            
            # Convert to float32 and normalize
            samples = samples.astype(np.float32)
            samples = samples / np.max(np.abs(samples))
            
            # If stereo, convert to mono
            if audio.channels == 2:
                samples = samples.reshape(-1, 2).mean(axis=1)
            
            print(f"Pydub success! Sample rate: {audio.frame_rate}Hz, Length: {len(samples)} samples")
            return samples, audio.frame_rate
        except Exception as e:
            print(f"Pydub loading failed: {str(e)}")

        # Try wave as last resort
        try:
            print("\nTrying Wave module...")
            with wave.open(audio_path, 'rb') as wav_file:
                # Get basic properties
                n_channels = wav_file.getnchannels()
                sampwidth = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                print(f"Wave file properties:")
                print(f"- Channels: {n_channels}")
                print(f"- Sample width: {sampwidth} bytes")
                print(f"- Frame rate: {framerate}Hz")
                print(f"- Number of frames: {n_frames}")
                
                # Read frames
                signal = wav_file.readframes(n_frames)
                signal = np.frombuffer(signal, dtype=np.int16)
                
                # Convert to float32 and normalize
                signal = signal.astype(np.float32) / 32768.0
                
                # If stereo, convert to mono
                if n_channels == 2:
                    signal = signal.reshape(-1, 2).mean(axis=1)
                
                print(f"Wave loading success! Length: {len(signal)} samples")
                return signal, framerate
        except Exception as e:
            print(f"Wave loading failed: {str(e)}")

        print("\nAll loading methods failed. Here's what we know about the file:")
        try:
            with open(audio_path, 'rb') as f:
                header = f.read(12)  # Read first 12 bytes
                print(f"First 12 bytes of file: {header.hex()}")
        except Exception as e:
            print(f"Couldn't read file header: {str(e)}")
            
        raise Exception("All audio loading methods failed")
    
    except Exception as e:
        raise Exception(f"Failed to load audio file {audio_path}: {str(e)}")

def convert_to_midi(audio_path, output_path, bpm=120):
    """
    Convert audio (vocals) to MIDI notes and save to output_path.
    """
    try:
        print("Starting MIDI conversion process...")
        print(f"Input audio: {audio_path}")
        print(f"Output MIDI: {output_path}")
        
        # Load and analyze audio using robust loader
        try:
            y, sr = load_audio(audio_path)
            print(f"Audio loaded successfully: {len(y)} samples at {sr}Hz")
        except Exception as e:
            raise Exception(f"Failed to load audio file: {str(e)}")
        
        try:
            # Perform pitch detection with optimized parameters
            print("Detecting pitch...")
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C6'),
                sr=sr,
                frame_length=2048,  # Larger frame for more stable pitch detection
                win_length=1024,    # Adjust window size
                hop_length=512      # Larger hop length for less dense notes
            )
            times = librosa.times_like(f0, sr=sr, hop_length=512)
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
            
            # Save MIDI file
            mid.save(output_path)
            print(f"MIDI file saved to: {output_path}")
            return output_path
            
        except Exception as e:
            raise Exception(f"Failed during pitch detection or MIDI creation: {str(e)}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise

# === Helper: Bandpass Filter for Vocoder ===
def bandpass_filter(signal, low, high, sr, order=4):
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
        
    b, a = butter(order, [low_normalized, high_normalized], btype='band')
    return lfilter(b, a, signal)

def get_envelope(signal, sr, cutoff=50.0):
    """Extract smooth envelope with lowpass filtering"""
    # Full-wave rectification
    rectified = np.abs(signal)
    
    # Low-pass filter design for envelope
    nyquist = 0.5 * sr
    cutoff_normalized = cutoff / nyquist
    cutoff_normalized = np.clip(cutoff_normalized, 0.001, 0.99)
    
    b, a = butter(4, cutoff_normalized, btype='low')
    
    # Apply filter and return
    return lfilter(b, a, rectified)

# === Vocoder-Style Processing ===
def run_vocoder(audio_path, output_wav, 
              formant_shift=1.0,      # 0.5-2.0: Lower values deepen voice, higher values raise it
              spectral_envelope=1.0,   # 0.5-2.0: Controls the overall spectral shape
              frequency_range=1.0,     # 0.5-2.0: Controls the frequency range of the vocoder
              aperiodicity=0.5,        # 0.0-1.0: Controls the amount of noise/breathiness
              clarity=0.7):            # 0.0-1.0: Controls the sharpness of formants
    """
    Enhanced vocoder implementation using WORLD for high-quality voice manipulation.
    Parameters control various aspects of the voice transformation.
    """
    print("Starting WORLD vocoder processing...")
    
    try:
        # Load the audio file
        voice, sr = load_audio(audio_path)
        print(f"Audio loaded: {len(voice)} samples at {sr}Hz")
        
        # WORLD works better with specific sample rates
        if sr != 16000:
            print(f"Resampling from {sr}Hz to 16000Hz for optimal processing...")
            voice = librosa.resample(voice, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Ensure audio is float64 (WORLD requirement)
        voice = voice.astype(np.float64)
        
        # Normalize input
        voice = voice / np.max(np.abs(voice))
        
        print("Extracting fundamental frequency (F0)...")
        # Extract pitch (F0) contour with more robust parameters
        _f0, t = pw.dio(voice, sr, f0_floor=50.0, f0_ceil=800.0)  # Wider F0 range
        f0 = pw.stonemask(voice, _f0, t, sr)  # Pitch refinement
        
        # Clean up F0 contour
        f0[f0 < 50.0] = 50.0  # Remove unrealistic low values
        f0[f0 > 800.0] = 800.0  # Remove unrealistic high values
        
        # Apply formant shift to F0 (with bounds checking)
        formant_shift = np.clip(formant_shift, 0.5, 2.0)
        f0 = f0 * formant_shift
        
        print("Extracting spectral envelope...")
        try:
            # Extract smoothed spectrogram with error handling
            sp = pw.cheaptrick(voice, f0, t, sr)
            if sp.shape[1] < 256:  # If spectrum is too small
                print("Warning: Spectrum too small, adjusting...")
                sp = np.pad(sp, ((0, 0), (0, 256 - sp.shape[1])), mode='edge')
        except Exception as e:
            print(f"Error in spectral extraction: {str(e)}")
            raise
        
        print("Extracting aperiodicity...")
        try:
            # Extract aperiodicity with error handling
            ap = pw.d4c(voice, f0, t, sr)
            if ap.shape[1] < 256:  # If aperiodicity is too small
                print("Warning: Aperiodicity too small, adjusting...")
                ap = np.pad(ap, ((0, 0), (0, 256 - ap.shape[1])), mode='edge')
        except Exception as e:
            print(f"Error in aperiodicity extraction: {str(e)}")
            raise
        
        # Ensure sp and ap have the same dimensions
        min_freq_len = min(sp.shape[1], ap.shape[1])
        sp = sp[:, :min_freq_len]
        ap = ap[:, :min_freq_len]
        
        # Modify spectral envelope (with bounds checking)
        spectral_envelope = np.clip(spectral_envelope, 0.5, 2.0)
        if spectral_envelope != 1.0:
            sp = np.power(sp, spectral_envelope)
        
        # Apply frequency range modification (with bounds checking)
        frequency_range = np.clip(frequency_range, 0.5, 2.0)
        if frequency_range != 1.0:
            print(f"Modifying frequency range with factor {frequency_range}")
            freq_len = sp.shape[1]
            new_freq_len = int(freq_len * frequency_range)
            
            # Ensure new_freq_len is within reasonable bounds
            new_freq_len = max(256, min(new_freq_len, min_freq_len))
            
            # Create temporary arrays for modified parameters
            sp_mod = np.zeros((sp.shape[0], new_freq_len))
            ap_mod = np.zeros((ap.shape[0], new_freq_len))
            
            # Interpolate each time frame
            x_orig = np.linspace(0, 1, freq_len)
            x_new = np.linspace(0, 1, new_freq_len)
            
            for i in range(sp.shape[0]):
                sp_mod[i] = np.interp(x_new, x_orig, sp[i])
                ap_mod[i] = np.interp(x_new, x_orig, ap[i])
            
            sp = sp_mod
            ap = ap_mod
        
        # Modify aperiodicity (with bounds checking)
        aperiodicity = np.clip(aperiodicity, 0.0, 1.0)
        ap = ap * aperiodicity
        
        print("Applying voice manipulation...")
        
        # Apply clarity enhancement (with bounds checking)
        clarity = np.clip(clarity, 0.0, 1.0)
        if clarity > 0.5:
            # Enhance formants by sharpening spectral peaks
            enhancement = (clarity - 0.5) * 2  # Map 0.5-1.0 to 0.0-1.0
            sp = np.power(sp, 1.0 + enhancement)
        else:
            # Smooth formants by softening spectral peaks
            smoothing = (0.5 - clarity) * 2  # Map 0.0-0.5 to 1.0-0.0
            kernel_size = int(smoothing * 10) * 2 + 1  # Odd number
            kernel_size = min(kernel_size, sp.shape[1] // 4)  # Prevent too large kernel
            if kernel_size > 1:
                for i in range(sp.shape[0]):
                    sp[i] = np.convolve(sp[i], np.ones(kernel_size)/kernel_size, mode='same')
        
        print("Synthesizing modified voice...")
        try:
            # Synthesize with modified parameters
            y = pw.synthesize(f0, sp, ap, sr)
            
            # Check if synthesis produced valid audio
            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                print("Warning: Invalid values in synthesis output, attempting to fix...")
                y = np.nan_to_num(y)  # Replace NaN/inf with valid values
            
            # Normalize output
            max_val = np.max(np.abs(y))
            if max_val > 0:
                y = y / max_val * 0.95
            else:
                raise Exception("Silent output detected")
            
        except Exception as e:
            print(f"Error in synthesis: {str(e)}")
            raise
        
        print("Applying final processing...")
        # Apply subtle bandpass filter to focus on voice frequencies
        y = bandpass_filter(y, 50, 7500, sr, order=4)
        
        print("Saving processed audio...")
        sf.write(output_wav, y, sr)
        print("Vocoder processing completed successfully")
        
    except Exception as e:
        print(f"Error during vocoder processing: {str(e)}")
        raise Exception(f"Vocoder processing failed: {str(e)}")

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

def run_talkbox(audio_path, output_wav, 
                drive=2.5,                    # Distortion amount (1.0-8.0)
                wah_min_freq=400,             # Minimum wah frequency
                wah_max_freq=2000,            # Maximum wah frequency
                wah_speed=3.0,                # Wah modulation speed in Hz
                wah_depth=0.8,                # Wah effect depth (0.0-1.0)
                wah_resonance=8.0,            # Wah filter resonance (Q factor)
                pre_emphasis_freq=1200,       # Pre-emphasis frequency
                compression_ratio=6.0,        # Compression ratio
                compression_attack=0.005,     # Compression attack time
                compression_release=0.15):    # Compression release time
    """
    Guitar-style talkbox implementation with enhanced wah effect
    """
    print("Starting guitar-style talkbox processing...")
    
    try:
        print("Importing required modules...")
        import soundfile as sf
        import numpy as np
        from scipy import signal
        print("Successfully imported modules")
            
        # Verify input file exists
        if not os.path.exists(audio_path):
            print(f"Error: Input file {audio_path} does not exist")
            return False
            
        print(f"Loading audio file: {audio_path}")
        info = sf.info(audio_path)
        print(f"File info: {info.samplerate}Hz, {info.channels} channels, {info.format} format, {info.subtype} subtype")
        
        try:
            audio_data, sample_rate = sf.read(audio_path)
            print(f"File loaded successfully. Shape: {audio_data.shape}")
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                print("Converting stereo to mono")
                audio_data = np.mean(audio_data, axis=1)
            
            # Convert to float32
            if audio_data.dtype != np.float32:
                print(f"Converting from {audio_data.dtype} to float32")
                audio_data = audio_data.astype(np.float32)
            
            # Normalize input
            max_val = np.abs(audio_data).max()
            if max_val > 1.0:
                print(f"Normalizing audio (max value was {max_val})")
                audio_data = audio_data / max_val
            
            print("Creating guitar-style filter chain...")
            try:
                nyquist = sample_rate / 2
                
                # Pre-emphasis filter (guitar pickup simulation)
                b_pre, a_pre = signal.butter(1, pre_emphasis_freq/nyquist, btype='high')
                
                # Initial bandpass to focus on guitar frequencies
                b_bandpass, a_bandpass = signal.butter(4, [80/nyquist, 6000/nyquist], btype='band')
                
                print("Processing audio with enhanced wah characteristics...")
                
                # Apply pre-emphasis
                audio_pre = signal.filtfilt(b_pre, a_pre, audio_data)
                
                # Apply initial bandpass
                filtered = signal.filtfilt(b_bandpass, a_bandpass, audio_pre)
                
                # Create time-varying wah filter
                t = np.arange(len(filtered)) / sample_rate
                
                # Generate wah frequency modulation (enhanced)
                wah_freq = wah_min_freq + (wah_max_freq - wah_min_freq) * \
                          (0.5 + 0.5 * np.sin(2 * np.pi * wah_speed * t))
                
                # Apply time-varying wah filter
                wahhed = filtered.copy()
                chunk_size = 2048  # Process in chunks for better control
                
                print("Applying wah effect...")
                for i in range(0, len(filtered), chunk_size):
                    chunk = filtered[i:i + chunk_size]
                    chunk_freq = wah_freq[i:i + chunk_size].mean()
                    b_wah, a_wah = create_wah_filter(chunk_freq, wah_resonance, sample_rate)
                    wahhed[i:i + chunk_size] = signal.filtfilt(b_wah, a_wah, chunk) * wah_depth + \
                                             chunk * (1 - wah_depth)
                
                # Add slight pitch modulation (vibrato)
                vibrato_rate = 6.0
                vibrato_depth = 0.001
                vibrato = 1.0 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t)
                modulated = wahhed * vibrato
                
                # Compression
                def compress_signal(x, threshold, ratio, attack_time, release_time, sample_rate):
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
                
                # Apply compression
                print("Applying compression and distortion...")
                compressed = compress_signal(modulated, 0.1, compression_ratio, 
                                          compression_attack, compression_release, sample_rate)
                
                # Apply tube-style distortion
                distorted = tube_distortion(compressed, drive=drive)
                
                # Final cleanup filter
                b_final, a_final = signal.butter(2, [60/nyquist, 7000/nyquist], btype='band')
                final = signal.filtfilt(b_final, a_final, distorted)
                
                # Normalize output
                final = final / np.max(np.abs(final)) * 0.95
                
                print(f"Processing complete. Output shape: {final.shape}")
                
                # Write output
                print(f"Writing output to {output_wav}")
                sf.write(output_wav, final, sample_rate, subtype='PCM_24')
                print("Audio successfully saved")
                return True
                
            except Exception as e:
                print(f"Error during processing: {str(e)}")
                print(f"Full error details: {type(e).__name__}")
                import traceback
                print(traceback.format_exc())
                return False
                
        except Exception as e:
            print(f"Error loading audio file: {str(e)}")
            print(f"Full error details: {type(e).__name__}")
            return False
        
    except Exception as e:
        print(f"Unexpected error in talkbox processing: {str(e)}")
        print(f"Full error details: {type(e).__name__}")
        import traceback
        print("Stack trace:")
        print(traceback.format_exc())
        return False
