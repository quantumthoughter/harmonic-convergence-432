import sys, os, json, uuid, math, subprocess
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['NUMBA_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['LIBROSA_CACHE_DIR'] = '/tmp/librosa_cache'
os.makedirs('/tmp/librosa_cache', exist_ok=True)
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

app = FastAPI(title="Harmonic Convergence · ∞ 432 Hz", version="2.2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

kernel = AtlanteanKernel()
engine = HarmonicEngine()
WORK_DIR = os.path.expanduser("~/Desktop/432_healed")
os.makedirs(WORK_DIR, exist_ok=True)

PYTESLA_PRESETS = {
    '440~432': {'left': 440.0, 'right': 432.0, 'beat': 8.0, 'label': '8 Hz (α/θ)'},
    '424~432': {'left': 424.0, 'right': 432.0, 'beat': 8.0, 'label': '8 Hz (α/θ)'},
    '440~444': {'left': 440.0, 'right': 444.0, 'beat': 4.0, 'label': '4 Hz (θ)'},
    '432~444': {'left': 432.0, 'right': 444.0, 'beat': 12.0, 'label': '12 Hz (β)'},
    '423~432': {'left': 423.0, 'right': 432.0, 'beat': 9.0, 'label': '9 Hz (α)'},
    '396~404': {'left': 396.0, 'right': 404.0, 'beat': 8.0, 'label': '8 Hz Engram'},
    '396~399': {'left': 396.0, 'right': 399.0, 'beat': 3.0, 'label': '3 Hz (δ)'},
}

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
    pytesla_enabled: bool = False
    pytesla_preset: str = '440~432'
    pytesla_swap: bool = False
    pytesla_swap_interval: float = 0.0
    carrier_33_5: float = 0.0
    carrier_7_83: float = 0.0
    carrier_8: float = 0.0

class DetectRequest(BaseModel): file_path: str
class ReportRequest(BaseModel): input_path: str; output_path: str
class BatchRequest(BaseModel): folder_path: str; output_dir: str; target_tuning: float = 432.0; format: str = "wav"
class SynthRequest(BaseModel): frequencies: dict = {}; duration_hours: float = 1.0; sample_rate: int = 48000
class RenderSynthRequest(BaseModel): frequencies: dict = {}; pyt_esla: bool = False; carrier_1hz: bool = True; ambience_mix: float = 0.0; duration_hours: float = 1.0; sample_rate: int = 48000

def source_name(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

def generate_carrier(duration, sr, freq, volume=0.0):
    if volume <= 0: return None
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    tone = np.sin(2 * np.pi * freq * t) * volume * 0.15
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t)
    stereo = np.zeros((2, n), dtype=np.float64)
    stereo[0] = tone * lfo; stereo[1] = tone * lfo
    return stereo

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
    atk = min(int(2.0 * sr), n); rel = min(int(4.0 * sr), n)
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

def apply_pytesla_swap(mix, sr, interval, left_tuning, right_tuning):
    if interval <= 0: return mix
    dur = mix.shape[-1] / sr
    n = mix.shape[-1]
    swap_count = int(dur / interval)
    if swap_count < 1: return mix
    seg_len = int(interval * sr)
    out = mix.copy()
    current_left = left_tuning
    current_right = right_tuning
    for i in range(min(swap_count, n // seg_len)):
        start = i * seg_len
        end = min((i + 1) * seg_len, n)
        if i % 2 == 1:
            out[0, start:end] = mix[1, start:end]
            out[1, start:end] = mix[0, start:end]
    return out

def heal_file(file_path, target_tuning, sr=None):
    session_id = uuid.uuid4().hex[:12]
    healed_path = os.path.join(WORK_DIR, f"{session_id}_healed.wav")
    heal_result = kernel.full_heal(file_path, healed_path, target_tuning=target_tuning)
    y, sr_orig = librosa.load(healed_path, sr=None, mono=False)
    sr = sr or sr_orig
    if sr != sr_orig:
        y = librosa.resample(y, orig_sr=sr_orig, target_sr=sr)
    if y.ndim == 1:
        stereo = np.zeros((2, len(y)), dtype=np.float64)
        stereo[0] = y; stereo[1] = y
    else:
        stereo = y.copy().astype(np.float64)
    os.remove(healed_path)
    return stereo, sr, heal_result

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
    if data[:4] != b'RIFF': return
    sname = title or os.path.splitext(os.path.basename(file_path))[0]
    list_data = bytearray(b'INFO')
    for ck_id, val in [('IART', artist), ('ICMP', composer), ('IPRD', album), ('ICMT', comment), ('INAM', sname)]:
        ck_val = val.encode('utf-8') + b'\x00' if isinstance(val, str) else val
        list_data += ck_id.encode('ascii') + struct.pack('<I', (len(ck_val) + 1) & ~1) + ck_val
        if len(ck_val) % 2: list_data += b'\x00'
    if bpm:
        bv = str(round(bpm, 1)).encode('utf-8') + b'\x00'
        list_data += b'ITMP' + struct.pack('<I', len(bv)) + bv
    if lyrics:
        lv = lyrics.encode('utf-8') + b'\x00'
        list_data += b'ILYR' + struct.pack('<I', len(lv)) + lv
    list_chunk = b'LIST' + struct.pack('<I', len(list_data)) + bytes(list_data)
    riff_size = len(data) - 8 + len(list_chunk)
    data[4:8] = struct.pack('<I', riff_size)
    new_data = bytes(data[:12]) + bytes(data[12:]) + list_chunk
    with open(file_path, 'wb') as f:
        f.write(new_data)

def write_metadata_mp3(file_path, artist="Quantum Thoughter", composer="Quantum Thoughter", album="Harmonic Convergence", comment="432 Hz Tuning Correction via Harmonic Convergence", bpm=None, title=None, lyrics=None):
    meta = ['-metadata', f'artist={artist}', '-metadata', f'composer={composer}', '-metadata', f'album={album}', '-metadata', f'comment={comment}']
    if title: meta += ['-metadata', f'title={title}']
    if bpm: meta += ['-metadata', f'tbpm={bpm}']
    if lyrics: meta += ['-metadata', f'lyrics={lyrics}']
    tmp_path = file_path + '.tmp.mp3'
    subprocess.run(['ffmpeg', '-y', '-i', file_path] + meta + ['-codec', 'copy', tmp_path], capture_output=True, timeout=120)
    if os.path.exists(tmp_path): os.replace(tmp_path, file_path)

def has_videotoolbox():
    try:
        r = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True, timeout=5)
        return 'videotoolbox' in r.stdout
    except: return False

def detect_hardware():
    info = {'cpu_count': os.cpu_count(), 'has_videotoolbox': has_videotoolbox()}
    try:
        r = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True, timeout=2)
        info['cpu'] = r.stdout.strip()
    except: info['cpu'] = 'unknown'
    try:
        r = subprocess.run(['system_profiler', 'SPDisplaysDataType'], capture_output=True, text=True, timeout=10)
        for line in r.stdout.split('\n'):
            if 'Chipset Model' in line or 'Metal' in line:
                info['metal'] = line.strip()
    except: pass
    return info

@app.get("/api/status")
def status():
    return {"status": "ready", "version": "2.2", "name": "Harmonic Convergence"}

@app.get("/api/hardware")
def hardware_info():
    return detect_hardware()

@app.get("/api/pytesla_presets")
def pytesla_presets():
    return PYTESLA_PRESETS

@app.post("/api/detect")
def detect_tuning(req: DetectRequest):
    if not os.path.exists(req.file_path):
        raise HTTPException(400, "File not found")
    result = kernel.detect_tuning(req.file_path)
    result['bpm'] = detect_bpm(req.file_path)
    return result

@app.post("/api/heal")
async def heal(request: HealRequest):
    if not os.path.exists(request.file_path):
        raise HTTPException(400, "File not found")
    session_id = uuid.uuid4().hex[:12]
    sname = source_name(request.file_path)
    sr = request.sample_rate
    is_pytesla = request.pytesla_enabled
    result_data = {}

    if is_pytesla:
        preset = PYTESLA_PRESETS.get(request.pytesla_preset, PYTESLA_PRESETS['440~432'])
        left_tuning = preset['left']
        right_tuning = preset['right']
        label = preset['label']
        beat = abs(left_tuning - right_tuning)

        heal_left_path = os.path.join(WORK_DIR, f"{session_id}_left.wav")
        heal_right_path = os.path.join(WORK_DIR, f"{session_id}_right.wav")

        cached = kernel.detect_tuning(request.file_path)
        try:
            left_result = kernel.full_heal(request.file_path, heal_left_path, target_tuning=left_tuning, cached_detect=cached)
            right_result = kernel.full_heal(request.file_path, heal_right_path, target_tuning=right_tuning, cached_detect=cached)
        except Exception as e:
            raise HTTPException(500, f"PyTesla healing failed: {e}")

        yL, srL = librosa.load(heal_left_path, sr=None, mono=False)
        yR, srR = librosa.load(heal_right_path, sr=None, mono=False)
        sr = sr or min(srL, srR)
        if srL != sr: yL = librosa.resample(yL, orig_sr=srL, target_sr=sr)
        if srR != sr: yR = librosa.resample(yR, orig_sr=srR, target_sr=sr)

        n = min(yL.shape[-1] if yL.ndim > 1 else len(yL), yR.shape[-1] if yR.ndim > 1 else len(yR))
        mix = np.zeros((2, n), dtype=np.float64)
        mix[0] = (yL[0] if yL.ndim > 1 else yL)[:n]
        mix[1] = (yR[0] if yR.ndim > 1 else yR)[:n]

        os.remove(heal_left_path)
        os.remove(heal_right_path)

        if request.pytesla_swap and request.pytesla_swap_interval > 0:
            mix = apply_pytesla_swap(mix, sr, request.pytesla_swap_interval, left_tuning, right_tuning)

        result_data = {
            'pytesla': True, 'left_tuning': left_tuning, 'right_tuning': right_tuning,
            'beat_hz': beat, 'beat_label': label,
            'left_result': {'original_tuning': left_result['original_tuning'], 'semitones_shifted': left_result['semitones_shifted']},
            'right_result': {'original_tuning': right_result['original_tuning'], 'semitones_shifted': right_result['semitones_shifted']},
        }
    else:
        healed_path = os.path.join(WORK_DIR, f"{session_id}_healed.wav")
        try:
            cached = kernel.detect_tuning(request.file_path)
            heal_result = kernel.full_heal(request.file_path, healed_path, target_tuning=request.target_tuning, cached_detect=cached)
        except Exception as e:
            raise HTTPException(500, f"Healing failed: {e}")
        y, sr_orig = librosa.load(healed_path, sr=None, mono=False)
        sr = request.sample_rate if request.sample_rate else sr_orig
        if sr != sr_orig:
            y = librosa.resample(y, orig_sr=sr_orig, target_sr=sr)
        if y.ndim == 1:
            mix = np.zeros((2, len(y)), dtype=np.float64)
            mix[0] = y; mix[1] = y
        else:
            mix = y.copy().astype(np.float64)
        os.remove(healed_path)
        result_data = {'heal_result': heal_result}

    final_wav = os.path.join(WORK_DIR, f"{session_id}_final.wav")
    dur = mix.shape[-1] / sr

    sub = generate_sub_bass(dur, sr, request.sub_bass, request.sub_bass_mode)
    if sub is not None:
        n = min(mix.shape[-1], sub.shape[-1])
        mix[..., :n] += sub[..., :n]

    c33 = generate_carrier(dur, sr, 33.5, request.carrier_33_5)
    if c33 is not None:
        n = min(mix.shape[-1], c33.shape[-1])
        mix[..., :n] += c33[..., :n]

    c783 = generate_carrier(dur, sr, 7.83, request.carrier_7_83)
    if c783 is not None:
        n = min(mix.shape[-1], c783.shape[-1])
        mix[..., :n] += c783[..., :n]

    c8 = generate_carrier(dur, sr, 8.0, request.carrier_8)
    if c8 is not None:
        n = min(mix.shape[-1], c8.shape[-1])
        mix[..., :n] += c8[..., :n]

    hc = generate_heart_coherence(dur, sr, request.heart_coherence)
    if hc is not None:
        n = min(mix.shape[-1], hc.shape[-1])
        mix[..., :n] += hc[..., :n]

    peak = np.max(np.abs(mix))
    if peak > 1.0: mix = mix / peak * 0.98
    sf.write(final_wav, mix.T, sr, subtype='PCM_16')
    bpm = detect_bpm(request.file_path)

    return {"session_id": session_id, "output_path": final_wav, "source_name": sname, "mix_shape": list(mix.shape), "sample_rate": sr, "bpm": bpm, **result_data}

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
        if cleanup == "1": _clean(session_id)
        return Response(content=data, media_type="audio/wav", headers={"Content-Disposition": f'attachment; filename="{safe_name}.wav"'})
    else:
        import subprocess, tempfile
        temp_mp3 = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        mp3_path = temp_mp3.name
        temp_mp3.close()
        meta = ['-metadata', f'artist={artist}', '-metadata', f'composer={composer}', '-metadata', f'album={album}', '-metadata', f'comment={comment}', '-metadata', f'title={fname}']
        if bpm_val: meta += ['-metadata', f'tbpm={bpm_val}']
        if lyr: meta += ['-metadata', f'lyrics={lyr}']
        if has_videotoolbox():
            subprocess.run(['ffmpeg', '-y', '-hwaccel', 'videotoolbox', '-i', wav_path, '-b:a', '320k', '-q:a', '0', '-joint_stereo', '1', '-id3v2_version', '3'] + meta + [mp3_path], capture_output=True, timeout=120)
        else:
            subprocess.run(['ffmpeg', '-y', '-i', wav_path, '-b:a', '320k', '-q:a', '0', '-joint_stereo', '1', '-id3v2_version', '3'] + meta + [mp3_path], capture_output=True, timeout=120)
        data = open(mp3_path, 'rb').read()
        os.remove(mp3_path)
        if cleanup == "1": _clean(session_id)
        return Response(content=data, media_type="audio/mpeg", headers={"Content-Disposition": f'attachment; filename="{safe_name}.mp3"'})

@app.post("/api/heal_batch")
def heal_batch(req: BatchRequest):
    import glob, time, concurrent.futures
    AUDIO_EXTS = {'.mp3','.wav','.flac','.ogg','.m4a','.aac'}
    files = sorted([f for f in glob.glob(os.path.join(req.folder_path, '*')) if os.path.splitext(f)[1].lower() in AUDIO_EXTS])
    if not files: raise HTTPException(400, "No audio files found in folder")
    os.makedirs(req.output_dir, exist_ok=True)
    kernel.clear_cache()
    start = time.time()
    max_workers = min(6, os.cpu_count() or 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as detect_ex:
        detect_futs = {detect_ex.submit(kernel.detect_tuning, f, False): f for f in files}
        detect_results = {}
        for fut in concurrent.futures.as_completed(detect_futs):
            f = detect_futs[fut]
            try: detect_results[f] = fut.result()
            except: detect_results[f] = None
    results = []
    is_mp3 = req.format == 'mp3'
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as heal_ex:
        heal_futs = {}
        for f in files:
            sname = source_name(f)
            out_path = os.path.join(req.output_dir, f"{sname}_∞432.wav")
            cached = detect_results.get(f)
            heal_futs[heal_ex.submit(kernel.full_heal, f, out_path, req.target_tuning, False, False, cached)] = (f, sname, cached)
        for fut in concurrent.futures.as_completed(heal_futs):
            f, sname, cached = heal_futs[fut]
            wav_path = os.path.join(req.output_dir, f"{sname}_∞432.wav")
            try:
                heal_result = fut.result()
                bpm = detect_bpm(f)
                verified_tuning = None
                if os.path.exists(wav_path):
                    try:
                        v = kernel.detect_tuning(wav_path, fast=False)
                        verified_tuning = v['tuning']
                    except: pass
                if is_mp3:
                    mp3_path = wav_path.rsplit('.', 1)[0] + '.mp3'
                    try:
                        meta = ['-metadata', f'artist=Quantum Thoughter', '-metadata', f'title={sname}', '-metadata', f'comment=432 Hz Tuning Correction via Harmonic Convergence']
                        if bpm: meta += ['-metadata', f'tbpm={bpm}']
                        subprocess.run(['ffmpeg', '-y', '-i', wav_path, '-b:a', '320k', '-q:a', '0', '-joint_stereo', '1', '-id3v2_version', '3'] + meta + [mp3_path], capture_output=True, timeout=120)
                        os.remove(wav_path)
                    except: pass
                    results.append({'file': os.path.basename(f), 'status': 'ok', 'output': os.path.basename(mp3_path), 'original_tuning': heal_result['original_tuning'], 'target_tuning': heal_result['target_tuning'], 'verified_tuning': verified_tuning, 'semitones_shifted': heal_result['semitones_shifted'], 'bpm': bpm})
                else:
                    try: write_metadata_wav(wav_path, title=sname, bpm=bpm)
                    except: pass
                    results.append({'file': os.path.basename(f), 'status': 'ok', 'output': os.path.basename(wav_path), 'original_tuning': heal_result['original_tuning'], 'target_tuning': heal_result['target_tuning'], 'verified_tuning': verified_tuning, 'semitones_shifted': heal_result['semitones_shifted'], 'bpm': bpm})
            except Exception as e:
                results.append({'file': os.path.basename(f), 'status': 'error', 'error': str(e)})
    total_time = time.time() - start
    verified_aligned = sum(1 for r in results if r.get('verified_tuning') and abs(r['verified_tuning'] - req.target_tuning) <= 1.0)
    return {'total': len(files), 'completed': sum(1 for r in results if r['status']=='ok'), 'failed': sum(1 for r in results if r['status']=='error'), 'verified_aligned': verified_aligned, 'time_seconds': round(total_time, 1), 'files_per_second': round(len(files)/total_time, 2) if total_time > 0 else 0, 'results': results}

@app.get("/api/batch_scan")
def batch_scan(folder_path: str):
    import glob
    AUDIO_EXTS = {'.mp3','.wav','.flac','.ogg','.m4a','.aac'}
    files = sorted([f for f in glob.glob(os.path.join(folder_path, '*')) if os.path.splitext(f)[1].lower() in AUDIO_EXTS])
    tuning = kernel.fast_detect_folder(folder_path, precise=False)
    file_list = []
    for f in files:
        d = tuning.get(f)
        if d:
            file_list.append({'file': os.path.basename(f), 'path': f, 'tuning': d['tuning'], 'confidence': d['confidence'], 'method': d['method']})
        else:
            file_list.append({'file': os.path.basename(f), 'path': f, 'tuning': None, 'confidence': 0, 'method': 'error'})
    return {'total': len(file_list), 'files': file_list}




@app.post("/api/report")
def generate_report(req: ReportRequest):
    if not os.path.exists(req.input_path) or not os.path.exists(req.output_path): raise HTTPException(400, "File not found")
    input_tuning = kernel.detect_tuning(req.input_path)['tuning']
    return kernel.generate_report(req.input_path, req.output_path, input_tuning)

@app.get("/api/verify")
def verify_folder(folder_path: str, target: float = 432.0):
    import glob, time
    AUDIO_EXTS = {'.mp3','.wav','.flac','.ogg','.m4a','.aac'}
    files = sorted([f for f in glob.glob(os.path.join(folder_path, '*')) if os.path.splitext(f)[1].lower() in AUDIO_EXTS])
    if not files: raise HTTPException(400, "No audio files found")
    start = time.time()
    results = []
    for f in files:
        try:
            d = kernel.detect_tuning(f, fast=False)
            deviation = round(d['tuning'] - target, 3)
            aligned = abs(deviation) <= 0.5
            results.append({'file': os.path.basename(f), 'tuning': d['tuning'], 'deviation_hz': deviation, 'confidence': round(d['confidence'], 3), 'aligned': aligned, 'method': d['method']})
        except Exception as e:
            results.append({'file': os.path.basename(f), 'tuning': None, 'deviation_hz': None, 'confidence': 0, 'aligned': False, 'error': str(e)})
    elapsed = round(time.time() - start, 1)
    aligned_count = sum(1 for r in results if r.get('aligned'))
    return {'total': len(results), 'aligned': aligned_count, 'misaligned': len(results) - aligned_count, 'target_hz': target, 'time_seconds': elapsed, 'results': results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
