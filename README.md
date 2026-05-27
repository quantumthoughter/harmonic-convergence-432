# 🌀 Resonance Sanctuary — 432 Hz Healing DAO

A sovereign audio healing workstation that re-anchors any music to 432 Hz using CQT-precision tuning detection, Rubberband pitch shift, and real-time A/B spectrum comparison.

## Quick Start
```bash
./launch.sh
```

## What It Does
1. **Drop any audio file** (MP3, WAV, FLAC, OGG)
2. **CQT detects the tuning** (440 Hz, 432 Hz, 444 Hz, etc.)
3. **Rubberband shifts pitch** to 432 Hz with zero quality loss
4. **Compare original vs healed** with side-by-side CQT spectrum analyzers
5. **Save as WAV or MP3** with correct metadata

## Architecture

### Backend (Python FastAPI)
| Endpoint | Description |
|----------|-------------|
| `GET /api/status` | Health check |
| `POST /api/detect` | CQT harmonic grid tuning detection |
| `POST /api/heal` | Rubberband pitch shift to target tuning |
| `GET /api/download/{id}?format=wav\|mp3` | Download healed file |

### Frontend (Electron)
- Pure 432 Hz healing pipeline with A/B comparison
- Dual CQT analyzers with 432 Hz grid line
- Diagnostic log with tuning shift, duration preservation
- Volume controls on both tracks
- Flower of Life animation during processing

### Audio Engine
- **librosa** for CQT analysis and pitch detection
- **Rubberband CLI** for high-quality pitch shifting
- **ffmpeg** for MP3 encoding with metadata
- **PCM_16** WAV output (standard, plays everywhere)

## Project Structure
```
432-healing-studio/
├── api_server.py              # Backend API server
├── backend/
│   ├── atlantean_kernel.py    # CQT detection + Rubberband shift
│   └── harmonic_engine.py     # Frequency layering engine
├── electron-frontend/
│   ├── index.html             # Main GUI
│   ├── main.js                # Electron main process
│   ├── preload.js             # Context bridge API
│   └── package.json
├── launch.sh                  # One-click launcher
├── requirements.txt
└── SESSION_SUMMARY.md         # Full development history
```

## Requirements
- Python 3.10+
- Node.js 18+
- Rubberband (`brew install rubberband`)
- ffmpeg (`brew install ffmpeg`)

## Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd electron-frontend && npm install && cd ..
```

## Development
```bash
./launch.sh                    # Start API + Electron
```

## License
MIT — free for healing, not for harm.
