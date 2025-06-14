interface TalkboxParams {
  // Guitar parameters
  drive: number;
  wah_min_freq: number;
  wah_max_freq: number;
  wah_speed: number;
  wah_depth: number;
  wah_resonance: number;
  pre_emphasis_freq: number;
  compression_ratio: number;
  compression_attack: number;
  compression_release: number;
  // Piano parameters
  hammer_hardness: number;
  string_resonance: number;
  brightness: number;
  sustain_level: number;
  room_size: number;
  damper_position: number;
  instrument_type: 'guitar' | 'piano';
}

interface VocoderParams {
  formant_shift: number;
  spectral_envelope: number;
  frequency_range: number;
  aperiodicity: number;
  clarity: number;
}

export async function processTalkbox(
  file: File,
  params: TalkboxParams,
  outputFormat: string = 'wav'
): Promise<Blob> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('params', JSON.stringify(params));
  formData.append('output_format', outputFormat);
  formData.append('type', 'talkbox');

  const response = await fetch('/api/process', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to process audio');
  }

  return response.blob();
}

export async function processVocoder(
  file: File,
  params: VocoderParams,
  outputFormat: string = 'wav'
): Promise<Blob> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('params', JSON.stringify(params));
  formData.append('output_format', outputFormat);
  formData.append('type', 'vocoder');

  const response = await fetch('/api/process', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to process audio');
  }

  return response.blob();
}

export async function processMidi(
  file: File,
  outputFormat: string = 'mid'
): Promise<Blob> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('output_format', outputFormat);
  formData.append('type', 'midi');

  const response = await fetch('/api/process', {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Failed to process audio');
  }

  return response.blob();
} 