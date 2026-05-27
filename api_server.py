import sys, os, json, uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import numpy as np
import soundfile as sf
import librosa

sys.path.insert(0, os.path.dirname(__file__))
from backend.atlantean_kernel import AtlanteanKernel
from backend.harmonic_engine import HarmonicEngine

app = FastAPI(title="432 Healing DAO API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

kernel = AtlanteanKernel()
engine = HarmonicEngine()
WORK_DIR = os.path.expanduser("~/Desktop/432_healed")
os.makedirs(WORK_DIR, exist_ok=True)

def src_name(path):
    return os.path.splitext(os.path.basename(path))[0]

class HealRequest(BaseModel):
    file_path: str
    target_tuning: float = 432.0
    reanchor: bool = False
    apply_eq: bool = False
    pure_432: bool = False
    sample_rate: Optional[int] = None
    solfeggio: dict = {}
    binaural_enabled: bool = False; binaural_left: float = 430.0; binaural_right: float = 434.0; binaural_volume: float = 0.15
    binaural_lfo: float = 0.0; binaural_hfo_rate: float = 0.0; binaural_hfo_depth: float = 0.0; binaural_pan: float = 0.0
    schumann_enabled: bool = False; schumann_volumes: dict = {}; stratosphere: bool = False
    heart_enabled: bool = False; breath_pacer: bool = True
    singularity_enabled: bool = False; singularity_base: float = 432.0; singularity_ratio: float = 1.5
    singularity_pulse: float = 0.1; singularity_depth: float = 0.3; singularity_mode: str = 'convergence'
    infrasound_enabled: bool = False; master_volume: float = 0.85

class DetectRequest(BaseModel): file_path: str
class ReportRequest(BaseModel): input_path: str; output_path: str
class SynthRequest(BaseModel): frequencies: dict = {}; duration_hours: float = 1.0; sample_rate: int = 48000
class RenderSynthRequest(BaseModel): frequencies: dict = {}; pyt_esla: bool = False; carrier_1hz: bool = True; ambience_mix: float = 0.0; duration_hours: float = 1.0; sample_rate: int = 48000

def source_name(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

@app.get("/api/status")
def status():
    return {"status": "ready", "version": "2.0"}

@app.post("/api/detect")
def detect_tuning(req: DetectRequest):
    if not os.path.exists(req.file_path):
        raise HTTPException(400, "File not found")
    return kernel.detect_tuning(req.file_path)

@app.post("/api/heal")
async def heal(request: HealRequest):
    if not os.path.exists(request.file_path):
        raise HTTPException(400, "File not found")
    session_id = uuid.uuid4().hex[:12]
    sname = source_name(request.file_path)
    healed_path = os.path.join(WORK_DIR, f"{session_id}_healed.wav")
    
    try:
        heal_result = kernel.full_heal(request.file_path, healed_path, target_tuning=request.target_tuning, reanchor=request.reanchor, apply_eq=request.apply_eq)
    except Exception as e:
        raise HTTPException(500, f"Healing failed: {e}")
    
    y, sr_orig = librosa.load(healed_path, sr=None, mono=False)
    sr = request.sample_rate if request.sample_rate else sr_orig
    if sr != sr_orig:
        y = librosa.resample(y, orig_sr=sr_orig, target_sr=sr)
    
    final_wav = os.path.join(WORK_DIR, f"{session_id}_final.wav")
    
    if request.pure_432:
        sf.write(final_wav, y.T, sr, subtype='PCM_16')
    else:
        config = {
            'master_volume': request.master_volume, 'solfeggio': request.solfeggio,
            'binaural': {'enabled': request.binaural_enabled, 'freq_left': request.binaural_left, 'freq_right': request.binaural_right, 'volume': request.binaural_volume, 'lfo_rate': request.binaural_lfo, 'hfo_rate': request.binaural_hfo_rate, 'hfo_depth': request.binaural_hfo_depth, 'crystal': True, 'pan': request.binaural_pan},
            'schumann': {'enabled': request.schumann_enabled, 'volumes': request.schumann_volumes, 'stratosphere': request.stratosphere},
            'heart_coherence': {'enabled': request.heart_enabled, 'breath_pacer': request.breath_pacer},
            'singularity': {'enabled': request.singularity_enabled, 'base_freq': request.singularity_base, 'ratio': request.singularity_ratio, 'pulse_rate': request.singularity_pulse, 'depth': request.singularity_depth, 'mode': request.singularity_mode},
            'infrasound': {'enabled': request.infrasound_enabled},
        }
        mix = engine.mix_all(y, sr, config)
        sf.write(final_wav, mix.T, sr, subtype='PCM_16')
    
    os.remove(healed_path)
    return {"session_id": session_id, "output_path": final_wav, "source_name": sname, "heal_result": heal_result, "mix_shape": list(y.shape if request.pure_432 else mix.shape), "sample_rate": sr}

@app.get("/api/download/{session_id}")
def download(session_id: str, format: str = "wav", name: str = "432_healed"):
    wav_path = os.path.join(WORK_DIR, f"{session_id}_final.wav")
    if not os.path.exists(wav_path):
        raise HTTPException(404, "Session not found")
    fname = name.replace('_432healed', '').replace('_432', '') + '_432'
    if format == "wav":
        return FileResponse(wav_path, media_type="audio/wav", filename=f"{fname}.wav")
    mp3_path = wav_path.replace('.wav', '.mp3')
    if not os.path.exists(mp3_path):
        import subprocess
        subprocess.run(['ffmpeg', '-y', '-i', wav_path, '-b:a', '320k', '-q:a', '0', '-joint_stereo', '1', '-id3v2_version', '3', '-metadata', f'title={fname}', '-metadata', 'artist=Resonance Sanctuary', '-metadata', 'comment=Healed to 432Hz via Resonance Sanctuary', mp3_path], capture_output=True, timeout=120)
    return FileResponse(mp3_path, media_type="audio/mpeg", filename=f"{fname}.mp3")

@app.post("/api/report")
def generate_report(req: ReportRequest):
    if not os.path.exists(req.input_path) or not os.path.exists(req.output_path):
        raise HTTPException(400, "File not found")
    input_tuning = kernel.detect_tuning(req.input_path)['tuning']
    return kernel.generate_report(req.input_path, req.output_path, input_tuning)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
