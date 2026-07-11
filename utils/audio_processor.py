import yt_dlp 
from pydub import AudioSegment
import os

DOWNLOAD_DIR = 'downloads'
os.makedirs(DOWNLOAD_DIR, exist_ok= True)

# youtube video to audio func.. using yt_dlp.

def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": False,

        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Sec-Fetch-Mode": "navigate",
        },
        # Rotate client players to find an unblocked stream type
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web", "mweb"],
                "skip": ["webpage", "hls"], # Skip metadata steps that flag bots
            }
        },
        "no_check_certificate": True, # Bypasses proxy-level validation issues
        "geo_bypass": True,           # Disables location mismatch tracking
    
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
    return filename


# data = download_youtube_audio("https://www.youtube.com/watch?v=B1ozfYMPNso")


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    if os.path.exists(output_path):
        return output_path
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000) #converting fule to 16kHz for whishper. 
    audio.export(output_path, format="wav")
    if "downloads" in input_path and input_path != output_path:
        try:
            os.remove(input_path)
        except Exception:
            pass
    return output_path

# data_final = convert_to_wav(data)


def chunk_audio(wav_path : str, chunk_minutes : int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000 

    chunks = []

    for i, start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path , format = "wav")

        chunks.append(chunk_path)
    
    return chunks
    
# print(chunk_audio(data_final))


def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        raw_path = download_youtube_audio(source)
        print("Standardizing YouTube audio to 16kHz Mono...")
        optimized_wav = convert_to_wav(raw_path)
    else:
        print("Detected local file. Processing and standardizing...")
        optimized_wav = convert_to_wav(source)

    print("Executing temporal slicing, Chunking...")
    chunks = chunk_audio(optimized_wav)
    print(f"Audio pipeline ingestion complete — {len(chunks)} chunk(s) generated.")
    return chunks