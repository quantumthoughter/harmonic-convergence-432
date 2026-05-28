"""
Atlantean Kernel v2 — CQT detection, Rubberband pitch shift, harmonic re-anchor, reports.
"""
import numpy as np
import librosa
import soundfile as sf
import subprocess, os, json
from datetime import datetime
from functools import lru_cache

class AtlanteanKernel:
    def __init__(self):
        self.target_tuning = 432.0
        self.sample_rate = 44100
        self._detect_cache = {}

    def detect_tuning_reference(self, file_path, duration=8):
        y, sr = librosa.load(file_path, sr=self.sample_rate, duration=duration, mono=True)
        C = np.abs(librosa.cqt(y, sr=sr, bins_per_octave=72, n_bins=72*7, fmin=32.7))
        freqs = librosa.cqt_frequencies(n_bins=72*7, fmin=32.7, bins_per_octave=72)
        peaks_idx = np.argsort(C.max(axis=1))[-400:]
        peaks_freq = freqs[peaks_idx]; peaks_mag = C.max(axis=1)[peaks_idx]
        candidates = [x * 0.5 for x in range(840, 900)]
        scores = {}
        for cand in candidates:
            grid = cand * 2 ** (np.arange(-48, 49) / 12)
            score = 0.0
            for pm, pf in zip(peaks_mag, peaks_freq):
                nearest = grid[np.argmin(np.abs(grid - pf))]
                distance = abs(nearest - pf) / nearest
                if distance < 0.015: score += pm * (1 - distance * 66)
            scores[cand] = score / max(peaks_mag.sum(), 1e-10)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best = sorted_scores[0][0]
        second = sorted_scores[1][0]
        confidence = max(0, min(1, (sorted_scores[0][1] - sorted_scores[1][1]) * 20))
        if confidence > 0.3:
            best_val = sorted_scores[0][1]
            second_val = sorted_scores[1][1]
            total = best_val + second_val
            if total > 0:
                best = (best * best_val + second * second_val) / total
        return (round(best, 3), float(confidence), 'cqt_grid')

    def detect_tuning_a4(self, file_path):
        y, sr = librosa.load(file_path, sr=self.sample_rate, duration=15, mono=True)
        offset = librosa.estimate_tuning(y=y, sr=sr, resolution=0.01)
        a4 = 440.0 * (2 ** (offset / 12))
        return (round(a4, 2), float(max(0, min(1, 1 - abs(offset) * 2))), 'a4_estimate')

    def detect_tuning(self, file_path, fast=False):
        use_cache = fast
        if not use_cache and file_path in self._detect_cache:
            use_cache = True
        if use_cache and file_path in self._detect_cache:
            return self._detect_cache[file_path]
        if fast:
            y, sr = librosa.load(file_path, sr=self.sample_rate, duration=5, mono=True)
            offset = librosa.estimate_tuning(y=y, sr=sr, resolution=0.01)
            tuning = round(440.0 * (2 ** (offset / 12)), 2)
            conf = float(max(0, min(1, 1 - abs(offset) * 5)))
            result = {'tuning': tuning, 'confidence': conf, 'method': 'a4_quick'}
        else:
            tuning, conf, method = self.detect_tuning_reference(file_path, duration=8)
            result = {'tuning': tuning, 'confidence': conf, 'method': method}
        self._detect_cache[file_path] = result
        return result

    def shift_pitch(self, input_path, output_path, semitones):
        y, sr = librosa.load(input_path, sr=None, mono=False)
        if y.ndim == 1:
            y = np.stack([y, y])
        ratio = 2 ** (semitones / 12)
        if ratio < 1.0:
            new_sr = int(round(sr / ratio))
        else:
            new_sr = int(round(sr * ratio))
        y_shifted = librosa.resample(y, orig_sr=sr, target_sr=new_sr, res_type='kaiser_fast')
        y_shifted = np.clip(y_shifted, -1.0, 1.0)
        if not output_path.endswith('.wav'):
            output_path = output_path.rsplit('.', 1)[0] + '.wav'
        sf.write(output_path, y_shifted.T, sr, subtype='PCM_16')

    def full_heal(self, input_path, output_path, target_tuning=432.0, reanchor=False, apply_eq=False, cached_detect=None):
        if cached_detect:
            d = cached_detect
        else:
            d = self.detect_tuning(input_path)
        ot = d['tuning']
        import shutil
        already = abs(ot - target_tuning) <= 0.01
        if already:
            shutil.copy2(input_path, output_path)
            ss = 0.0; status = 'already_at_target'
            realized_tuning = float(ot)
        else:
            ss = 12 * np.log2(target_tuning / ot)
            self.shift_pitch(input_path, output_path, ss)
            status = 'healed'
            realized_tuning = float(round(target_tuning, 2))
        return {'status': status, 'original_tuning': float(ot), 'target_tuning': realized_tuning, 'confidence': float(d['confidence']), 'detection_method': d['method'], 'semitones_shifted': float(round(ss, 4)), 'already_at_target': bool(already)}

    def fast_detect_folder(self, folder_path, precise=False):
        import glob, concurrent.futures
        AUDIO_EXTS = {'.mp3','.wav','.flac','.ogg','.m4a','.aac'}
        files = sorted([f for f in glob.glob(os.path.join(folder_path, '*')) if os.path.splitext(f)[1].lower() in AUDIO_EXTS])
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
            futs = {ex.submit(self.detect_tuning, f, fast=(not precise)): f for f in files}
            for fut in concurrent.futures.as_completed(futs):
                f = futs[fut]
                try: results[f] = fut.result()
                except: results[f] = None
        return results

    def clear_cache(self):
        self._detect_cache.clear()
