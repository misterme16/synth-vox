from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import tempfile
import os
import shutil
from processing_modules import TalkboxEffect, VocoderEffect

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TalkboxParams(BaseModel):
    # Guitar parameters
    drive: Optional[float] = 2.5
    wah_min_freq: Optional[float] = 400
    wah_max_freq: Optional[float] = 2000
    wah_speed: Optional[float] = 3.0
    wah_depth: Optional[float] = 0.8
    wah_resonance: Optional[float] = 8.0
    pre_emphasis_freq: Optional[float] = 1200
    compression_ratio: Optional[float] = 6.0
    compression_attack: Optional[float] = 0.005
    compression_release: Optional[float] = 0.15
    # Piano parameters
    hammer_hardness: Optional[float] = 0.7
    string_resonance: Optional[float] = 0.8
    brightness: Optional[float] = 0.6
    sustain_level: Optional[float] = 0.7
    room_size: Optional[float] = 0.5
    damper_position: Optional[float] = 0.8
    instrument_type: Optional[str] = 'guitar'

class VocoderParams(BaseModel):
    formant_shift: Optional[float] = 1.0
    spectral_envelope: Optional[float] = 1.0
    frequency_range: Optional[float] = 1.0
    aperiodicity: Optional[float] = 0.5
    clarity: Optional[float] = 0.7

@app.post("/process/talkbox")
async def process_talkbox(
    file: UploadFile = File(...),
    params: str = Form(...),
    output_format: str = Form(default="wav")
):
    try:
        # Parse parameters
        params_dict = TalkboxParams.parse_raw(params).dict()
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded file
            input_path = os.path.join(temp_dir, "input.wav")
            with open(input_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Process audio
            output_path = os.path.join(temp_dir, f"output.{output_format}")
            processor = TalkboxEffect()
            processor.process_audio(input_path, output_path, params_dict)
            
            # Return processed file
            return FileResponse(
                output_path,
                media_type=f"audio/{output_format}",
                filename=f"processed.{output_format}"
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/process/vocoder")
async def process_vocoder(
    file: UploadFile = File(...),
    params: str = Form(...),
    output_format: str = Form(default="wav")
):
    try:
        # Parse parameters
        params_dict = VocoderParams.parse_raw(params).dict()
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded file
            input_path = os.path.join(temp_dir, "input.wav")
            with open(input_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Process audio
            output_path = os.path.join(temp_dir, f"output.{output_format}")
            processor = VocoderEffect()
            processor.process_audio(input_path, output_path, params_dict)
            
            # Return processed file
            return FileResponse(
                output_path,
                media_type=f"audio/{output_format}",
                filename=f"processed.{output_format}"
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/process/midi")
async def process_midi(
    file: UploadFile = File(...),
    output_format: str = Form(default="mid")
):
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded file
            input_path = os.path.join(temp_dir, "input.wav")
            with open(input_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Process audio to MIDI
            output_path = os.path.join(temp_dir, f"output.{output_format}")
            processor = TalkboxEffect()
            processor.audio_to_midi(input_path, output_path)
            
            # Return processed file
            return FileResponse(
                output_path,
                media_type="audio/midi",
                filename=f"processed.{output_format}"
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        ) 