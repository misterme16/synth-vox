import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get('file') as File;
    const type = formData.get('type') as string;
    const params = JSON.parse(formData.get('params') as string);

    if (!file || !type) {
      return NextResponse.json(
        { error: 'Missing required fields' },
        { status: 400 }
      );
    }

    // Convert file to base64
    const buffer = await file.arrayBuffer();
    const base64Audio = Buffer.from(buffer).toString('base64');

    // Call the FastAPI backend
    const response = await fetch('http://localhost:8000/api/process', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        audio: base64Audio,
        type: type,
        params: params,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to process audio');
    }

    const result = await response.json();

    // Convert base64 back to audio blob
    const processedAudioBuffer = Buffer.from(result.audio, 'base64');
    
    // Set content type and filename based on processing type
    const contentType = type === 'midi' ? 'audio/midi' : 'audio/wav';
    const extension = type === 'midi' ? '.mid' : '.wav';
    
    // Return the processed file
    return new NextResponse(processedAudioBuffer, {
      headers: {
        'Content-Type': contentType,
        'Content-Disposition': `attachment; filename="processed_${file.name.split('.')[0]}${extension}"`,
      },
    });
  } catch (error) {
    console.error('Error processing audio:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Failed to process audio' },
      { status: 500 }
    );
  }
} 