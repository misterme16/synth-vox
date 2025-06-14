import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'edge'; // Use edge runtime for better performance

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    const params = formData.get('params') as string;
    const outputFormat = formData.get('output_format') as string || 'wav';
    const processingType = formData.get('type') as string || 'talkbox';

    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      );
    }

    // Convert File to ArrayBuffer for Python API
    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    // Create FormData for Python API
    const pythonFormData = new FormData();
    pythonFormData.append('file', new Blob([buffer], { type: file.type }), file.name);
    pythonFormData.append('params', params);
    pythonFormData.append('output_format', outputFormat);

    // Determine the API endpoint based on processing type
    const apiEndpoint = processingType === 'midi' 
      ? '/api/process/midi'
      : processingType === 'vocoder'
      ? '/api/process/vocoder'
      : '/api/process/talkbox';

    // Call Python API
    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || ''}${apiEndpoint}`, {
      method: 'POST',
      body: pythonFormData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to process audio');
    }

    // Get the processed audio data
    const processedAudio = await response.blob();

    // Return the processed audio
    return new NextResponse(processedAudio, {
      headers: {
        'Content-Type': `audio/${outputFormat}`,
        'Content-Disposition': `attachment; filename="processed.${outputFormat}"`,
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

// Configure CORS
export async function OPTIONS() {
  return new NextResponse(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      'Access-Control-Max-Age': '86400',
    },
  });
} 