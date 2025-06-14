import argparse
import json
import sys
import os

# Add the parent directory to the Python path to import the processing modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))

from src.talkbox import TalkboxEffect
from src.vocoder import Vocoder
from src.midi_converter import MidiConverter

def main():
    parser = argparse.ArgumentParser(description='Process audio files')
    parser.add_argument('--input', required=True, help='Input audio file path')
    parser.add_argument('--output', required=True, help='Output audio file path')
    parser.add_argument('--type', required=True, choices=['talkbox', 'vocoder', 'midi'], help='Processing type')
    parser.add_argument('--params', required=True, help='Processing parameters as JSON string')
    
    args = parser.parse_args()
    params = json.loads(args.params)

    try:
        if args.type == 'talkbox':
            processor = TalkboxEffect(
                drive=params.get('drive', 2.5),
                wah_min_freq=params.get('wahMinFreq', 400),
                wah_max_freq=params.get('wahMaxFreq', 2000),
                wah_speed=params.get('wahSpeed', 3.0),
                wah_depth=params.get('wahDepth', 0.8),
                wah_resonance=params.get('wahResonance', 8.0),
                pre_emphasis_freq=params.get('preEmphasisFreq', 1200),
                compression_ratio=params.get('compressionRatio', 6.0),
                compression_attack=params.get('compressionAttack', 0.005),
                compression_release=params.get('compressionRelease', 0.15)
            )
            processor.process_file(args.input, args.output)

        elif args.type == 'vocoder':
            processor = Vocoder(
                formant_shift=params.get('formantShift', 1.0),
                spectral_envelope=params.get('spectralEnvelope', 1.0),
                frequency_range=params.get('frequencyRange', 1.0),
                aperiodicity=params.get('aperiodicity', 0.5),
                clarity=params.get('clarity', 0.7)
            )
            processor.process_file(args.input, args.output)

        elif args.type == 'midi':
            processor = MidiConverter()
            processor.convert_to_midi(args.input, args.output)

    except Exception as e:
        print(f"Error processing audio: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main() 