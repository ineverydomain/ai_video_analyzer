AI Video & Meeting Assistant 🎙️🎥

A production-ready AI-powered meeting analytics and Video RAG (Retrieval-Augmented Generation) assistant. This application processes local audio/video files or YouTube video URLs, standardizes and chunks the audio streams, orchestrates multi-engine transcription (via local Whisper or cloud APIs), and provides automated insight extraction (Action Items, Key Decisions, Open Questions) along with a fully interactive Q&A chatbot to "chat with your meeting."

🚀 Architectural Overview

The application features a decoupled architecture separating media processing, speech-to-text pipeline execution, linguistic insight extraction, and vector databases.

                  ┌──────────────────────┐
                  │  Audio/Video Source  │
                  │ (Local File / URL)   │
                  └──────────┬───────────┘
                             │
                             ▼
               ┌───────────────────────────┐
               │    Audio Processing       │
               │ (yt_dlp, pydub, ffmpeg)   │
               └─────────────┬─────────────┘
                             │ (16kHz Mono WAV Chunks)
                             ▼
               ┌───────────────────────────┐
               │   Dual Compute Routing    │
               └──────┬─────────────┬──────┘
                      │             │
        ┌─────────────▼─────┐ ┌─────▼─────────────────────────┐
        │  Local CPU Mode   │ │        Cloud API Mode         │
        ├───────────────────┤ ├───────────────────────────────┤
        │ • Whisper (Local) │ │ • Groq API (Whisper-v3-large)  │
        │ • HF Embeddings   │ │ • Sarvam AI (Hinglish Trans)  │
        │   (all-MiniLM-L6) │ │ • OpenAI Embeddings           │
        │ • Ollama (llama2) │ │ • OpenAI Chat (gpt-4o-mini)   │
        └─────────────┬─────┘ └─────┬─────────────────────────┘
                      │             │
                      ▼             ▼
        ┌───────────────────┐ ┌───────────────────────────────┐
        │  Chroma DB Local  │ │        Chroma DB Cloud        │
        │  (384 dimensions) │ │       (1536 dimensions)       │
        └─────────────┬─────┘ └─────┬─────────────────────────┘
                      │             │
                      └──────┬──────┘
                             ▼
               ┌───────────────────────────┐
               │   Insights & RAG Engine   │
               │ (Summary, Decisions, QA)  │
               └───────────────────────────┘

1. Ingestion & Preprocessing

Source Agnosticism: Automatically detects source inputs (URLs vs. Local files).

Audio Standardization: Leverages yt_dlp for extracting clean audio from streaming links and converts files using pydub/FFmpeg to standard 16kHz Mono WAV format required by industry-grade Speech-to-Text architectures.

Temporal Slicing: Automatically segments audio into manageable chunks to respect cloud payload boundaries (e.g., Sarvam's 30-second execution limit).

2. Dual Compute Environments

Cloud API (Fast): Leverages cloud services including Groq (Whisper-Large-v3) for instant English transcriptions, Sarvam AI for complex multi-lingual Hinglish translation, and OpenAI embeddings (text-embedding-3-small / 1536-dim) paired with GPT-4o-mini. This environment is lightweight and lightning-fast.

Local CPU (Private): Dynamically builds an isolated on-device machine learning environment running OpenAI's Whisper model locally along with HuggingFace open-source transformers (all-MiniLM-L6-v2 / 384-dim) for high-privacy deployments.

3. Dynamic Vector Database Routing

To prevent database corruption and collision errors, the RAG engine automatically partitions database files using separate collection paths:

Local Mode: meeting_transcript_local (matching the 384-dimension embeddings from all-MiniLM-L6-v2).

API Mode: meeting_transcript_api (matching the 1536-dimension embeddings from OpenAI).

🛠️ Tech Stack & Dependencies

Core Orchestration: LangChain (LCEL Expression Language, Runnables, Core Prompts)

Frontend UI: Streamlit Framework

Vector Engine: Chroma DB (LangChain Integration)

Media Processing: Pydub, FFmpeg, YT-DLP

AI Models Supported: OpenAI GPT-4o-mini, Groq Cloud Whisper-v3, Sarvam AI Speech Translate, Local OpenAI Whisper, HuggingFace Transformers

📂 Project Structure

├── app.py # Streamlit Main UI Application File
├── main.py # Command-Line Interface (CLI) Engine Pipeline
├── requirements.txt # Production Python Dependencies Package Manifest
├── .gitignore # Strict Source Exclusion Mask File
├── core/
│ ├── transcriber.py # Core Audio routing engine (Local/Groq/Sarvam)
│ ├── extractor.py # Insight analyst (Action Items, Decisions, Questions)
│ ├── summarizer.py # Map-Reduce structural text summarizing engine
│ ├── rag_engine.py # LCEL QA Pipeline assembly and context loading
│ └── vector_store.py # Chromadb configuration & vector routing factory
└── utils/
└── audio_processor.py # Audio downsampling, mono mixing & temporal slicing

⚙️ Setup & Installation Instructions

1. Prerequisites

Ensure you have Python 3.9+ installed and FFmpeg configured on your local system path.

Windows (PowerShell Admin): choco install ffmpeg or download directly from the official FFmpeg site and add to your System PATH variables.

Mac (Homebrew): brew install ffmpeg

Linux: sudo apt install ffmpeg

2. Clone and Setup Environment

# Clone the repository

git clone [https://github.com/ineverydomain/ai_video_analyzer.git](https://github.com/ineverydomain/ai_video_analyzer.git)
cd ai_video_analyzer

# Create a virtual environment

python -m venv venv
source venv/bin/activate # On Windows use: venv\Scripts\activate

# Install required packages

pip install -r requirements.txt

3. Setup Secrets Config (.env)

Create a .env file in the root directory for your local executions (Note: This file is blocked from GitHub tracking by the .gitignore configuration):

OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here
WHISPER_MODEL=tiny

💻 Running the Application

Method A: Launching Streamlit Interface (Recommended)

streamlit run app.py

Open your browser to the designated localhost port (typically http://localhost:8501). Configure settings on the left sidebar panel to switch between environments.

Method B: Executing Terminal CLI Engine

python main.py

☁️ Cloud Deployment Protocol (Streamlit Community Cloud)

Connect your GitHub repository to your Streamlit Account.

Sign in to your portal dashboard at share.streamlit.io.

Create a New App, select your target repository/branch, and point to app.py as the main launch track.

Open Advanced settings -> Navigate to Secrets and drop in the production API key blocks using valid TOML formatting:

OPENAI*API_KEY = "sk-..."
GROQ_API_KEY = "gsk*..."
SARVAM_API_KEY = "..."

Click Deploy.

💡 Pro-Tip for Hosting: For public hosted environments (like Streamlit Cloud's free tier), running under Cloud API Mode is strongly recommended. Attempting to download and load a local Whisper model or Hugging Face embedding pipeline on-the-fly will exceed the platform container's 1GB RAM limit and instantly crash the environment container.

🔒 Security & Git Management

This project enforces strict Git parameters. The .gitignore is pre-configured to suppress cache binaries, local Whisper models, temporary .wav sliced audio elements, local Chroma database fragments, and private .env environment records to prevent credential exposure.

To clear Git's cache and apply the ignores cleanly, run:

git rm -r --cached .
git add .
git commit -m "chore: apply gitignore strictly"
