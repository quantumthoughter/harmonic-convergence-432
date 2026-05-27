import sys, os, json, uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from urllib.parse import quote
from pydantic import BaseModel
from typing import Optional
import numpy as np
import soundfile as sf
import librosa

sys.path.insert(0, os.path.dirname(__file__))
from backend.atlantean_kernel import AtlanteanKernel
from backend.harmonic_engine import HarmonicEngine

app = FastAPI(title="Harmonic Convergence · ∞ 432 Hz", version="2.1")
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
    pure_432: bool = True
    sample_rate: Optional[int] = None
    solfeggio: dict = {}
    binaural_enabled: bool = False; binaural_left: float = 430.0; binaural_right: float = 434.0; binaural_volume: float = 0.15
    binaural_lfo: float = 0.0; binaural_hfo_rate: float = 0.0; binaural_hfo_depth: float = 0.0; binaural_pan: float = 0.0
    schumann_enabled: bool = False; schumann_volumes: dict = {}; stratosphere: bool = False
    heart_enabled: bool = False; breath_pacer: bool = True
    singularity_enabled: bool = False; singularity_base: float = 432.0; singularity_ratio: float = 1.5
    singularity_pulse: float = 0.1; singularity_depth: float = 0.3; singularity_mode: str = 'convergence'
    infrasound_enabled: bool = False; master_volume: float = 0.85
    sub_bass: float = 0.0
    sub_bass_mode: str = 'full'
    heart_coherence: float = 0.0

class DetectRequest(BaseModel): file_path: str
class ReportRequest(BaseModel): input_path: str; output_path: str
class BatchRequest(BaseModel): folder_path: str; output_dir: str; target_tuning: float = 432.0
class SynthRequest(BaseModel): frequencies: dict = {}; duration_hours: float = 1.0; sample_rate: int = 48000
class RenderSynthRequest(BaseModel): frequencies: dict = {}; pyt_esla: bool = False; carrier_1hz: bool = True; ambience_mix: float = 0.0; duration_hours: float = 1.0; sample_rate: int = 48000

def source_name(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

def generate_sub_bass(duration, sr, volume=0.0, mode='full'):
    if volume <= 0: return None
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    bands = {
        'full': [13.5, 27.0, 33.5, 54.0, 72.0, 108.0, 216.0, 432.0],
        'gamma': [33.5, 72.0, 108.0],
        'delta': [13.5, 27.0, 54.0],
        'theta': [54.0, 72.0, 108.0],
        'sub': [27.0, 33.5, 54.0],
    }
    freqs = bands.get(mode, bands['full'])
    layers = []
    for f in freqs:
        phase = np.random.random() * 2 * np.pi
        tone = np.sin(2 * np.pi * f * t + phase)
        layers.append(tone)
    mix = sum(layers) / len(layers)
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t)
    breath_env = np.ones(n)
    atk = min(int(2.0 * sr), n)
    rel = min(int(4.0 * sr), n)
    breath_env[:atk] = np.linspace(0, 1, atk)
    breath_env[-rel:] = np.linspace(1, 0, rel)
    mix = mix * lfo * breath_env * volume * 0.25
    stereo = np.zeros((2, n), dtype=np.float64)
    stereo[0] = mix * 0.6; stereo[1] = mix * 0.6
    return stereo

def generate_heart_coherence(duration, sr, volume=0.0):
    if volume <= 0: return None
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    tone = np.sin(2 * np.pi * 1.0 * t)
    breath = 0.6 + 0.4 * np.sin(2 * np.pi * 0.1 * t)
    tone = tone * breath * volume * 0.12
    stereo = np.zeros((2, n), dtype=np.float64)
    stereo[0] = tone; stereo[1] = tone
    return stereo

def detect_bpm(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=30, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return round(float(tempo), 1)
    except:
        return None

def write_metadata_wav(file_path, artist="Quantum Thoughter", composer="Quantum Thoughter", album="Harmonic Convergence", comment="432 Hz Tuning Correction via Harmonic Convergence", bpm=None, title=None, lyrics=None):
    import struct
    with open(file_path, 'rb') as f:
        data = bytearray(f.read())
    if data[:4] != b'RIFF':
        return
    sname = (title or os.path.splitext(os.path.basename(file_path))[0])
    list_data = bytearray(b'INFO')
    for ck_id, val in [('IART', artist), ('ICMP', composer), ('IPRD', album), ('ICMT', comment), ('INAM', sname)]:
        ck_val = val.encode('utf-8') + b'\x00' if isinstance(val, str) else val
        list_data += ck_id.encode('ascii') + struct.pack('<I', len(ck_val)) + ck_val
    if bpm:
        ck_val = str(round(bpm, 1)).encode('utf-8') + b'\x00'
        list_data += b'ITMP' + struct.pack('<I', len(ck_val)) + ck_val
    if lyrics:
        ck_val = lyrics.encode('utf-8') + b'\x00'
        list_data += b'ILYR' + struct.pack('<I', len(ck_val)) + ck_val
    list_chunk = b'LIST' + struct.pack('<I', len(list_data)) + bytes(list_data)
    new_data = bytes(data[:12]) + list_chunk + bytes(data[12:])
    with open(file_path, 'wb') as f:
        f.write(new_data)

def write_metadata_mp3(file_path, artist="Quantum Thoughter", composer="Quantum Thoughter", album="Harmonic Convergence", comment="432 Hz Tuning Correction via Harmonic Convergence", bpm=None, title=None, lyrics=None):
    import subprocess
    meta = [
        '-metadata', f'artist={artist}',
        '-metadata', f'composer={composer}',
        '-metadata', f'album={album}',
        '-metadata', f'comment={comment}',
    ]
    if title:
        meta += ['-metadata', f'title={title}']
    if bpm:
        meta += ['-metadata', f'tbpm={bpm}']
    if lyrics:
        meta += ['-metadata', f'lyrics={lyrics}']
        meta += ['-metadata', f'uslt={lyrics}']
    tmp_path = file_path + '.tmp.mp3'
    subprocess.run(['ffmpeg', '-y', '-i', file_path] + meta + ['-codec', 'copy', tmp_path], capture_output=True, timeout=120)
    if os.path.exists(tmp_path):
        os.replace(tmp_path, file_path)

@app.get("/api/status")
def status():
    return {"status": "ready", "version": "2.1", "name": "Harmonic Convergence"}

@app.post("/api/detect")
def detect_tuning(req: DetectRequest):
    if not os.path.exists(req.file_path):
        raise HTTPException(400, "File not found")
    result = kernel.detect_tuning(req.file_path)
    bpm = detect_bpm(req.file_path)
    result['bpm'] = bpm
    return result

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
    if y.ndim == 1:
        mix = np.zeros((2, len(y)), dtype=np.float64)
        mix[0] = y; mix[1] = y
    else:
        mix = y.copy().astype(np.float64)
    dur = mix.shape[-1] / sr
    sub = generate_sub_bass(dur, sr, request.sub_bass, request.sub_bass_mode)
    if sub is not None:
        n = min(mix.shape[-1], sub.shape[-1])
        mix[..., :n] += sub[..., :n]
    hc = generate_heart_coherence(dur, sr, request.heart_coherence)
    if hc is not None:
        n = min(mix.shape[-1], hc.shape[-1])
        mix[..., :n] += hc[..., :n]
    peak = np.max(np.abs(mix))
    if peak > 1.0:
        mix = mix / peak * 0.98
    sf.write(final_wav, mix.T, sr, subtype='PCM_16')
    os.remove(healed_path)
    bpm = detect_bpm(request.file_path)
    return {"session_id": session_id, "output_path": final_wav, "source_name": sname, "heal_result": heal_result, "mix_shape": list(mix.shape), "sample_rate": sr, "bpm": bpm}

def _clean_old(age_hours=1):
    import time, glob
    now = time.time()
    for f in glob.glob(os.path.join(WORK_DIR, "*_final.*")):
        try:
            if now - os.path.getmtime(f) > age_hours * 3600: os.remove(f)
        except: pass

def _clean(session_id):
    import glob
    for f in glob.glob(os.path.join(WORK_DIR, f"{session_id}_*")):
        try: os.remove(f)
        except: pass

@app.get("/api/download/{session_id}")
def download(session_id: str, format: str = "wav", name: str = "432_healed", cleanup: str = "1", artist: str = "Quantum Thoughter", composer: str = "Quantum Thoughter", album: str = "Harmonic Convergence", comment: str = "432 Hz Tuning Correction via Harmonic Convergence", bpm: str = "", lyrics: str = ""):
    wav_path = os.path.join(WORK_DIR, f"{session_id}_final.wav")
    if not os.path.exists(wav_path):
        raise HTTPException(404, "Session not found")
    fname = os.path.basename(name)
    safe_name = quote(fname, safe='')
    bpm_val = float(bpm) if bpm else None
    lyr = lyrics if lyrics else None
    if format == "wav":
        write_metadata_wav(wav_path, artist=artist, composer=composer, album=album, comment=comment, bpm=bpm_val, title=fname, lyrics=lyr)
        data = open(wav_path, 'rb').read()
        if cleanup == "1":
            _clean(session_id)
        return Response(content=data, media_type="audio/wav", headers={"Content-Disposition": f'attachment; filename="{safe_name}.wav"'})
    else:
        import subprocess, tempfile
        temp_mp3 = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        mp3_path = temp_mp3.name
        temp_mp3.close()
        meta = ['-metadata', f'artist={artist}', '-metadata', f'composer={composer}', '-metadata', f'album={album}', '-metadata', f'comment={comment}', '-metadata', f'title={fname}']
        if bpm_val:
            meta += ['-metadata', f'tbpm={bpm_val}']
        if lyr:
            meta += ['-metadata', f'lyrics={lyr}']
        subprocess.run(['ffmpeg', '-y', '-i', wav_path, '-b:a', '320k', '-q:a', '0', '-joint_stereo', '1', '-id3v2_version', '3'] + meta + [mp3_path], capture_output=True, timeout=120)
        data = open(mp3_path, 'rb').read()
        os.remove(mp3_path)
        if cleanup == "1":
            _clean(session_id)
        return Response(content=data, media_type="audio/mpeg", headers={"Content-Disposition": f'attachment; filename="{safe_name}.mp3"'})

@app.post("/api/heal_batch")
def heal_batch(req: BatchRequest):
    import glob
    AUDIO_EXTS = {'.mp3','.wav','.flac','.ogg','.m4a','.aac'}
    files = [f for f in glob.glob(os.path.join(req.folder_path, '*')) if os.path.splitext(f)[1].lower() in AUDIO_EXTS]
    if not files:
        raise HTTPException(400, "No audio files found in folder")
    os.makedirs(req.output_dir, exist_ok=True)
    results = []
    for f in files:
        try:
            sname = source_name(f)
            out_path = os.path.join(req.output_dir, f"{sname}_∞432.wav")
            heal_result = kernel.full_heal(f, out_path, target_tuning=req.target_tuning)
            bpm = detect_bpm(f)
            write_metadata_wav(out_path, title=sname, bpm=bpm)
            results.append({'file': os.path.basename(f), 'status': 'ok', 'output': os.path.basename(out_path), 'original_tuning': heal_result['original_tuning'], 'target_tuning': heal_result['target_tuning'], 'semitones_shifted': heal_result['semitones_shifted'], 'bpm': bpm})
        except Exception as e:
            results.append({'file': os.path.basename(f), 'status': 'error', 'error': str(e)})
    return {'total': len(files), 'completed': sum(1 for r in results if r['status']=='ok'), 'failed': sum(1 for r in results if r['status']=='error'), 'results': results}

@app.post("/api/report")
def generate_report(req: ReportRequest):
    if not os.path.exists(req.input_path) or not os.path.exists(req.output_path):
        raise HTTPException(400, "File not found")
    input_tuning = kernel.detect_tuning(req.input_path)['tuning']
    return kernel.generate_report(req.input_path, req.output_path, input_tuning)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
