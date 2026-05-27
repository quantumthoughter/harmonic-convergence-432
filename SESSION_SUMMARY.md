# 🌀 Resonance Sanctuary — Session History

## Architecture
- **Backend:** Python FastAPI on port 8765 (`api_server.py`)
- **Frontend:** Electron app (`electron-frontend/`)
- **Audio Engine:** Rubberband CLI for pitch shift, librosa for analysis
- **CQT Analysis:** Constant-Q Transform with logarithmic frequency mapping

## Core Pipeline
1. **Drop audio** → `/api/detect` → CQT harmonic grid analysis finds tuning
2. **HEAL** → `/api/heal` → Rubberband shifts to 432 Hz → saves 16-bit PCM WAV
3. **A/B Comparison** → Two CQT analyzers side by side (original vs healed)
4. **Save** → Downloads WAV or MP3 with correct metadata (`Song_432.wav` / `Song_432.mp3`)

## File Structure
```
432-healing-studio/
├── api_server.py              ← FastAPI backend (all endpoints)
├── backend/
│   ├── atlantean_kernel.py    ← CQT detection + Rubberband shift
│   └── harmonic_engine.py     ← Frequency layering (solfeggio, binaural, etc.)
├── electron-frontend/
│   ├── index.html             ← Main GUI (pure 432 pipeline + A/B CQT)
│   ├── main.js                ← Electron window + IPC handlers
│   ├── preload.js             ← Context bridge API
│   └── package.json
├── launch.sh                  ← Starts API + Electron
├── presets/
│   └── research_presets.json
├── requirements.txt
└── SESSION_SUMMARY.md          ← This file
```

## Key Endpoints
- `GET /api/status` — Health check
- `POST /api/detect` — CQT tuning detection
- `POST /api/heal` — Pitch shift to target tuning
- `GET /api/download/{id}?format=wav|mp3&name=file_name` — Download healed file
- `POST /api/report` — Spectral comparison report

## Current State
- ✅ CQT detection works (440 Hz → 432 Hz verified)
- ✅ Rubberband pitch shift works (-0.318 semitones)
- ✅ A/B comparison with dual CQT analyzers
- ✅ WAV/MP3 save with correct naming
- ❌ Session files accumulate in `~/Desktop/432_healed/` — cleanup runs on files >1hr old
- ✅ 432 Hz grid line on CQT is accurate
- ✅ Volume controls on both tracks
- ✅ Flower of Life animation during processing

## Known Issues
- `~/Desktop/432_healed/` accumulates session files (`_final.wav`, `_final.mp3`)
- CQT needs audio playing to show spectrum (idle shows baseline noise)
- Electron contextIsolation prevents `file://` fetch from working in some cases

## Three Next Logics (Pending)
1. **Automation Timeline** — keyframe-based frequency/volume/pan changes
2. **Metatron 12/13 Hz** — cube frequency generator
3. **Nakshatra Presets** — astrology-based frequency sets

## Commands
```bash
./launch.sh                    # Start everything
source venv/bin/activate       # Activate Python env
python api_server.py           # Start API only
cd electron-frontend && npx electron .  # Start Electron only
```
