import os
from logging import getLogger

import ffmpeg

logger = getLogger(__name__)

def create_low_quality_wav(input_wav_path):
    """
    Creates a low-quality version of a WAV file with 1000Hz sampling rate.
    
    Args:
        input_wav_path: Path to the input WAV file
    
    Returns:
        Path to the created low-quality WAV file
    """
    try:
        # Get the directory and filename
        directory = os.path.dirname(input_wav_path)
        filename = os.path.basename(input_wav_path)
        
        # Create output filename
        base_name = os.path.splitext(filename)[0]
        low_quality_filename = f"compressed_{base_name}.mp3"
        low_quality_path = os.path.join(directory, low_quality_filename)
        
        # Create compressed version with better quality
        stream = ffmpeg.input(input_wav_path)
        stream = ffmpeg.output(
            stream, 
            low_quality_path,
            acodec='libmp3lame',  # MP3 codec offers good compression/quality ratio
            ac=2,                 # Explicitly maintain 2 channels (stereo)
            ar=22050,             # 22.05kHz sampling rate (good balance of quality/size)
            ab='64k',             # 64kbps bitrate (adjust as needed)
            q=9                   # Quality setting for MP3 (lower is better, range: 0-9)
        )
        ffmpeg.run(stream, quiet=True, overwrite_output=True)
        
        logger.info(f"Created compressed version of audio at {low_quality_path}")
        return low_quality_path
        
    except Exception as e:
        logger.error(f"Failed to create compressed audio version: {e}")
        return None

if __name__ == "__main__":
    # Use the specified file
    input_file = "./calls/test.wav"
    
    # Ensure the input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.")
    else:
        output_path = create_low_quality_wav(input_file)
        if output_path:
            print(f"Successfully created compressed version at: {output_path}")
