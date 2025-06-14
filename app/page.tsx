'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

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

export default function Home() {
  const [selectedTab, setSelectedTab] = useState(0);
  const [processing, setProcessing] = useState(false);
  const [inputFile, setInputFile] = useState<File | null>(null);
  const [processingType, setProcessingType] = useState('talkbox');
  const [isDragging, setIsDragging] = useState(false);
  const [instrumentType, setInstrumentType] = useState<'guitar' | 'piano'>('guitar');

  // Talkbox parameters
  const [talkboxParams, setTalkboxParams] = useState<TalkboxParams>({
    // Guitar parameters
    drive: 2.5,
    wah_min_freq: 400,
    wah_max_freq: 2000,
    wah_speed: 3.0,
    wah_depth: 0.8,
    wah_resonance: 8.0,
    pre_emphasis_freq: 1200,
    compression_ratio: 6.0,
    compression_attack: 0.005,
    compression_release: 0.15,
    // Piano parameters
    hammer_hardness: 0.7,
    string_resonance: 0.8,
    brightness: 0.6,
    sustain_level: 0.7,
    room_size: 0.5,
    damper_position: 0.8,
    instrument_type: 'guitar'
  });

  // Vocoder parameters
  const [vocoderParams, setVocoderParams] = useState<VocoderParams>({
    formant_shift: 1.0,
    spectral_envelope: 1.0,
    frequency_range: 1.0,
    aperiodicity: 0.5,
    clarity: 0.7
  });

  const talkboxPresets: Record<string, TalkboxParams> = {
    // Guitar presets
    metal: {
      drive: 6.0,
      pre_emphasis_freq: 1500,
      compression_ratio: 8.0,
      wah_resonance: 10.0,
      wah_depth: 0.7,
      wah_min_freq: 400,
      wah_max_freq: 2000,
      wah_speed: 3.0,
      compression_attack: 0.005,
      compression_release: 0.15,
      hammer_hardness: 0.7,
      string_resonance: 0.8,
      brightness: 0.6,
      sustain_level: 0.7,
      room_size: 0.5,
      damper_position: 0.8,
      instrument_type: 'guitar'
    },
    blues: {
      drive: 3.0,
      pre_emphasis_freq: 1000,
      wah_min_freq: 300,
      wah_max_freq: 2500,
      compression_ratio: 5.0,
      wah_resonance: 6.0,
      wah_depth: 0.6,
      wah_speed: 2.0,
      compression_attack: 0.01,
      compression_release: 0.2,
      hammer_hardness: 0.7,
      string_resonance: 0.8,
      brightness: 0.6,
      sustain_level: 0.7,
      room_size: 0.5,
      damper_position: 0.8,
      instrument_type: 'guitar'
    },
    jazz: {
      drive: 1.5,
      pre_emphasis_freq: 800,
      wah_depth: 0.5,
      compression_ratio: 4.0,
      wah_resonance: 6.0,
      wah_min_freq: 200,
      wah_max_freq: 1500,
      wah_speed: 1.5,
      compression_attack: 0.015,
      compression_release: 0.25,
      hammer_hardness: 0.7,
      string_resonance: 0.8,
      brightness: 0.6,
      sustain_level: 0.7,
      room_size: 0.5,
      damper_position: 0.8,
      instrument_type: 'guitar'
    },
    wah: {
      drive: 4.0,
      wah_min_freq: 200,
      wah_max_freq: 3000,
      wah_speed: 4.0,
      wah_depth: 0.9,
      wah_resonance: 12.0,
      pre_emphasis_freq: 1200,
      compression_ratio: 6.0,
      compression_attack: 0.005,
      compression_release: 0.15,
      hammer_hardness: 0.7,
      string_resonance: 0.8,
      brightness: 0.6,
      sustain_level: 0.7,
      room_size: 0.5,
      damper_position: 0.8,
      instrument_type: 'guitar'
    },
    // Piano presets
    concert_grand: {
      drive: 2.5,
      wah_min_freq: 400,
      wah_max_freq: 2000,
      wah_speed: 3.0,
      wah_depth: 0.8,
      wah_resonance: 8.0,
      pre_emphasis_freq: 1200,
      hammer_hardness: 0.7,
      string_resonance: 0.9,
      brightness: 0.7,
      sustain_level: 0.8,
      room_size: 0.7,
      damper_position: 0.8,
      compression_ratio: 4.0,
      compression_attack: 0.01,
      compression_release: 0.2,
      instrument_type: 'piano'
    },
    upright: {
      drive: 2.5,
      wah_min_freq: 400,
      wah_max_freq: 2000,
      wah_speed: 3.0,
      wah_depth: 0.8,
      wah_resonance: 8.0,
      pre_emphasis_freq: 1200,
      hammer_hardness: 0.8,
      string_resonance: 0.7,
      brightness: 0.6,
      sustain_level: 0.6,
      room_size: 0.4,
      damper_position: 0.7,
      compression_ratio: 5.0,
      compression_attack: 0.008,
      compression_release: 0.18,
      instrument_type: 'piano'
    },
    bright_pop: {
      drive: 2.5,
      wah_min_freq: 400,
      wah_max_freq: 2000,
      wah_speed: 3.0,
      wah_depth: 0.8,
      wah_resonance: 8.0,
      pre_emphasis_freq: 1200,
      hammer_hardness: 0.9,
      string_resonance: 0.6,
      brightness: 0.9,
      sustain_level: 0.5,
      room_size: 0.3,
      damper_position: 0.6,
      compression_ratio: 6.0,
      compression_attack: 0.005,
      compression_release: 0.15,
      instrument_type: 'piano'
    },
    warm_jazz: {
      drive: 2.5,
      wah_min_freq: 400,
      wah_max_freq: 2000,
      wah_speed: 3.0,
      wah_depth: 0.8,
      wah_resonance: 8.0,
      pre_emphasis_freq: 1200,
      hammer_hardness: 0.6,
      string_resonance: 0.8,
      brightness: 0.5,
      sustain_level: 0.7,
      room_size: 0.6,
      damper_position: 0.9,
      compression_ratio: 3.0,
      compression_attack: 0.015,
      compression_release: 0.25,
      instrument_type: 'piano'
    }
  };

  const vocoderPresets = {
    original: {
      formant_shift: 1.0,
      spectral_envelope: 1.0,
      frequency_range: 1.0,
      aperiodicity: 0.5,
      clarity: 0.7
    },
    robot: {
      formant_shift: 1.0,
      spectral_envelope: 1.2,
      frequency_range: 1.5,
      aperiodicity: 0.8,
      clarity: 0.9
    },
    alien: {
      formant_shift: 1.8,
      spectral_envelope: 1.5,
      frequency_range: 1.8,
      aperiodicity: 0.6,
      clarity: 0.5
    },
    deep: {
      formant_shift: 0.85,
      spectral_envelope: 1.1,
      frequency_range: 0.9,
      aperiodicity: 0.4,
      clarity: 0.75
    },
    high: {
      formant_shift: 1.5,
      spectral_envelope: 0.8,
      frequency_range: 1.2,
      aperiodicity: 0.4,
      clarity: 0.9
    }
  };

  // Function to check if parameters match a preset
  const isMatchingPreset = (preset: any, currentParams: any) => {
    return Object.entries(preset).every(([key, value]) => {
      // Use a small epsilon for floating point comparison
      const epsilon = 0.001;
      return Math.abs(currentParams[key] - (value as number)) < epsilon;
    });
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setInputFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setInputFile(e.dataTransfer.files[0]);
    }
  };

  const handleInstrumentChange = (type: 'guitar' | 'piano') => {
    setInstrumentType(type);
    setTalkboxParams(prev => ({
      ...prev,
      instrument_type: type
    }));
  };

  const handleProcess = async () => {
    if (!inputFile) return;

    setProcessing(true);
    try {
      const formData = new FormData();
      formData.append('file', inputFile);
      formData.append('type', processingType);
      formData.append('params', JSON.stringify(
        processingType === 'talkbox' ? talkboxParams : vocoderParams
      ));

      const response = await fetch('/api/process', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to process audio');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `processed_${inputFile.name}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      // Show success message
      alert('Audio processed successfully!');
    } catch (error) {
      console.error('Error processing audio:', error);
      alert(error instanceof Error ? error.message : 'Failed to process audio');
    } finally {
      setProcessing(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      <div className="container mx-auto px-4 py-12">
        <motion.h1 
          className="text-5xl font-bold text-center mb-8 bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-pink-600"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          Synth-Vox
        </motion.h1>
        
        <motion.div 
          className="max-w-4xl mx-auto"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <div className="bg-gray-800 rounded-xl shadow-2xl p-8 mb-8 backdrop-blur-lg bg-opacity-50">
            <div className="space-y-6">
              {/* File Upload Area */}
              <div 
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors duration-200 ${
                  isDragging 
                    ? 'border-purple-500 bg-purple-500 bg-opacity-10' 
                    : 'border-gray-600 hover:border-purple-500'
                }`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <div className="flex flex-col items-center space-y-4">
                  <svg className="w-12 h-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  <div className="text-lg">
                    {inputFile ? (
                      <span className="text-purple-400">{inputFile.name}</span>
                    ) : (
                      <span>Drag & drop your audio file or click to browse</span>
                    )}
                  </div>
                  <input
                    type="file"
                    accept={processingType === 'midi' ? '.wav,.mp3' : '.wav,.mp3,.mid'}
                    onChange={handleFileChange}
                    className="hidden"
                    id="file-upload"
                  />
                  <label
                    htmlFor="file-upload"
                    className="px-4 py-2 bg-purple-600 rounded-lg cursor-pointer hover:bg-purple-700 transition-colors duration-200"
                  >
                    Choose File
                  </label>
                </div>
              </div>

              {/* Processing Type Selection */}
              <div className="space-y-2">
                <label className="text-sm font-medium text-gray-300">
                  Processing Type
                </label>
                <div className="grid grid-cols-3 gap-4">
                  {['talkbox', 'vocoder', 'midi'].map((type) => (
                    <button
                      key={type}
                      onClick={() => setProcessingType(type)}
                      className={`p-4 rounded-lg transition-all duration-200 ${
                        processingType === type
                          ? 'bg-purple-600 shadow-lg scale-105'
                          : 'bg-gray-700 hover:bg-gray-600'
                      }`}
                    >
                      {type.charAt(0).toUpperCase() + type.slice(1)}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {processingType === 'talkbox' && (
            <div className="bg-gray-800 rounded-xl shadow-2xl p-8 mb-8 backdrop-blur-lg bg-opacity-50">
              <div className="space-y-4">
                <label className="text-sm font-medium text-gray-300">
                  Instrument Type
                </label>
                <div className="grid grid-cols-2 gap-4">
                  <button
                    onClick={() => handleInstrumentChange('guitar')}
                    className={`p-4 rounded-lg transition-all duration-200 ${
                      instrumentType === 'guitar'
                        ? 'bg-purple-600 shadow-lg scale-105'
                        : 'bg-gray-700 hover:bg-gray-600'
                    }`}
                  >
                    Guitar
                  </button>
                  <button
                    onClick={() => handleInstrumentChange('piano')}
                    className={`p-4 rounded-lg transition-all duration-200 ${
                      instrumentType === 'piano'
                        ? 'bg-purple-600 shadow-lg scale-105'
                        : 'bg-gray-700 hover:bg-gray-600'
                    }`}
                  >
                    Piano
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Parameters Section */}
          <AnimatePresence mode="wait">
            {processingType !== 'midi' && (
              <motion.div
                key="parameters"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.3 }}
                className="bg-gray-800 rounded-xl shadow-2xl p-8 mb-8 backdrop-blur-lg bg-opacity-50"
              >
                <div className="flex justify-between mb-6">
                  <button
                    className={`px-6 py-2 rounded-lg transition-colors duration-200 ${
                      selectedTab === 0
                        ? 'bg-purple-600'
                        : 'bg-gray-700 hover:bg-gray-600'
                    }`}
                    onClick={() => setSelectedTab(0)}
                  >
                    Parameters
                  </button>
                  <button
                    className={`px-6 py-2 rounded-lg transition-colors duration-200 ${
                      selectedTab === 1
                        ? 'bg-purple-600'
                        : 'bg-gray-700 hover:bg-gray-600'
                    }`}
                    onClick={() => setSelectedTab(1)}
                  >
                    Presets
                  </button>
                </div>

                <AnimatePresence mode="wait">
                  {selectedTab === 0 ? (
                    <motion.div
                      key="params"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: 20 }}
                      transition={{ duration: 0.2 }}
                      className="space-y-6"
                    >
                      {processingType === 'talkbox' ? (
                        <>
                          {instrumentType === 'guitar' ? (
                            // Guitar parameters
                            Object.entries(talkboxParams)
                              .filter(([key]) => !key.toString().includes('piano') && key !== 'instrument_type')
                              .map(([key, value]) => (
                                <div key={key} className="space-y-2">
                                  <div className="flex justify-between">
                                    <label className="text-sm font-medium text-gray-300">
                                      {key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
                                    </label>
                                    <span className="text-purple-400">{(value as number).toFixed(3)}</span>
                                  </div>
                                  <input
                                    type="range"
                                    min={key.toString().includes('attack') || key.toString().includes('release') ? 0.001 : 1.0}
                                    max={key.toString().includes('freq') ? 2000 : key.toString().includes('ratio') ? 20.0 : 8.0}
                                    step={key.toString().includes('attack') || key.toString().includes('release') ? 0.001 : 0.1}
                                    value={value as number}
                                    onChange={(e) => setTalkboxParams({ ...talkboxParams, [key]: parseFloat(e.target.value) })}
                                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                                  />
                                </div>
                              ))
                          ) : (
                            // Piano parameters
                            <>
                              {(([
                                ['hammer_hardness', 'Hammer Hardness', 0, 1],
                                ['string_resonance', 'String Resonance', 0, 1],
                                ['brightness', 'Brightness', 0, 1],
                                ['sustain_level', 'Sustain Level', 0, 1],
                                ['room_size', 'Room Size', 0, 1],
                                ['damper_position', 'Damper Position', 0, 1],
                                ['compression_ratio', 'Compression Ratio', 1, 20],
                                ['compression_attack', 'Compression Attack', 0.001, 0.1],
                                ['compression_release', 'Compression Release', 0.01, 1]
                              ] as const).map(([paramKey, label, min, max]) => {
                                const paramValue = talkboxParams[paramKey as keyof TalkboxParams];
                                return (
                                  <div key={paramKey} className="space-y-2">
                                    <div className="flex justify-between">
                                      <label className="text-sm font-medium text-gray-300">
                                        {label}
                                      </label>
                                      <span className="text-purple-400">
                                        {typeof paramValue === 'number' ? paramValue.toFixed(3) : paramValue}
                                      </span>
                                    </div>
                                    <input
                                      type="range"
                                      min={min}
                                      max={max}
                                      step={paramKey.includes('attack') || paramKey.includes('release') ? 0.001 : 0.1}
                                      value={paramValue}
                                      onChange={(e) => setTalkboxParams({ ...talkboxParams, [paramKey]: parseFloat(e.target.value) })}
                                      className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                                    />
                                  </div>
                                );
                              }))}
                            </>
                          )}
                        </>
                      ) : (
                        <>
                          {Object.entries(vocoderParams).map(([key, value]) => (
                            <div key={key} className="space-y-2">
                              <div className="flex justify-between">
                                <label className="text-sm font-medium text-gray-300">
                                  {key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
                                </label>
                                <span className="text-purple-400">{value.toFixed(2)}</span>
                              </div>
                              <input
                                type="range"
                                min={0.0}
                                max={2.0}
                                step={0.1}
                                value={value}
                                onChange={(e) => setVocoderParams({ ...vocoderParams, [key]: parseFloat(e.target.value) })}
                                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                              />
                            </div>
                          ))}
                        </>
                      )}
                    </motion.div>
                  ) : (
                    <motion.div
                      key="presets"
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      transition={{ duration: 0.2 }}
                      className="grid grid-cols-2 md:grid-cols-3 gap-4"
                    >
                      {processingType === 'talkbox'
                        ? Object.entries(talkboxPresets)
                            .filter(([_, preset]) => preset.instrument_type === instrumentType)
                            .map(([name, preset]) => (
                              <button
                                key={name}
                                onClick={() => setTalkboxParams(preset)}
                                className={`p-4 rounded-lg transition-all duration-200 ${
                                  isMatchingPreset(preset, talkboxParams)
                                    ? 'bg-purple-600 shadow-lg scale-105'
                                    : 'bg-gray-700 hover:bg-gray-600'
                                }`}
                              >
                                {name.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
                              </button>
                            ))
                        : Object.entries(vocoderPresets).map(([name, preset]) => (
                            <button
                              key={name}
                              onClick={() => setVocoderParams(preset)}
                              className={`p-4 rounded-lg transition-all duration-200 ${
                                isMatchingPreset(preset, vocoderParams)
                                  ? 'bg-purple-600 shadow-lg scale-105'
                                  : 'bg-gray-700 hover:bg-gray-600'
                              }`}
                            >
                              {name.charAt(0).toUpperCase() + name.slice(1)}
                            </button>
                          ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Process Button */}
          <div className="flex justify-center">
            <motion.button
              onClick={handleProcess}
              disabled={!inputFile || processing}
              className={`px-8 py-4 rounded-lg text-lg font-medium transition-all duration-200 ${
                !inputFile || processing
                  ? 'bg-gray-600 cursor-not-allowed'
                  : 'bg-gradient-to-r from-purple-600 to-pink-600 hover:scale-105'
              }`}
              whileHover={{ scale: !inputFile || processing ? 1 : 1.05 }}
              whileTap={{ scale: !inputFile || processing ? 1 : 0.95 }}
            >
              {processing ? (
                <div className="flex items-center space-x-2">
                  <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  <span>Processing...</span>
                </div>
              ) : (
                'Process Audio'
              )}
            </motion.button>
          </div>
        </motion.div>
      </div>
    </main>
  );
}
