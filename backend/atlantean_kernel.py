"""
Atlantean Kernel v2 — CQT detection, Rubberband pitch shift, harmonic re-anchor, reports.
"""
import numpy as np
import librosa
import soundfile as sf
import subprocess, os, json
from datetime import datetime

class AtlanteanKernel:
    def __init__(self):
        self.target_tuning = 432.0
        self.sample_rate = 44100

    def detect_tuning_reference(self, file_path):
        y, sr = librosa.load(file_path, sr=self.sample_rate, duration=15, mono=True)
        C = np.abs(librosa.cqt(y, sr=sr, bins_per_octave=36, n_bins=36*7, fmin=32.7))
        freqs = librosa.cqt_frequencies(n_bins=36*7, fmin=32.7, bins_per_octave=36)
        peaks_idx = np.argsort(C.max(axis=1))[-200:]
        peaks_freq = freqs[peaks_idx]; peaks_mag = C.max(axis=1)[peaks_idx]
        candidates = [420,427,430,432,435,438,440,442,444,445,446,448]
        scores = {}
        for cand in candidates:
            grid = cand * 2 ** (np.arange(-48, 49) / 12)
            score = 0.0
            for pm, pf in zip(peaks_mag, peaks_freq):
                nearest = grid[np.argmin(np.abs(grid - pf))]
                distance = abs(nearest - pf) / nearest
                if distance < 0.03: score += pm * (1 - distance * 33.3)
            scores[cand] = score / max(peaks_mag.sum(), 1e-10)
        best = max(scores, key=scores.get)
        second = sorted(scores, key=scores.get, reverse=True)[1]
        confidence = max(0, min(1, (scores[best] - scores[second]) * 10))
        return (best, float(confidence), 'cqt_grid')

    def detect_tuning_a4(self, file_path):
        y, sr = librosa.load(file_path, sr=self.sample_rate, duration=15, mono=True)
        offset = librosa.estimate_tuning(y=y, sr=sr, resolution=0.01)
        a4 = 440.0 * (2 ** (offset / 12))
        return (round(a4, 2), float(max(0, min(1, 1 - abs(offset) * 2))), 'a4_estimate')

    def detect_tuning(self, file_path):
        tuning, conf, method = self.detect_tuning_reference(file_path)
        if conf < 0.3 and method == 'cqt_grid':
            a4_t, a4_c, _ = self.detect_tuning_a4(file_path)
            if a4_c > conf: tuning, conf, method = a4_t, a4_c, 'a4_estimate'
        return {'tuning': tuning, 'confidence': conf, 'method': method}

    def shift_pitch(self, input_path, output_path, semitones):
        result = subprocess.run(["rubberband", "-t", "1.0", "-p", str(semitones), input_path, output_path], capture_output=True, text=True)
        if result.returncode != 0: raise RuntimeError(f"Rubberband failed: {result.stderr}")

    def full_heal(self, input_path, output_path, target_tuning=432.0, reanchor=False, apply_eq=False):
        d = self.detect_tuning(input_path)
        ot = d['tuning']
        y_orig, sr = librosa.load(input_path, sr=self.sample_rate)
        od = len(y_orig) / sr
        already = abs(ot - target_tuning) <= 1.5
        if already:
            import shutil; shutil.copy2(input_path, output_path)
            ss = 0.0; status = 'already_at_target'
        else:
            ss = 12 * np.log2(target_tuning / ot)
            self.shift_pitch(input_path, output_path, ss)
            status = 'healed'
        y_healed, _ = librosa.load(output_path, sr=self.sample_rate)
        hd = len(y_healed) / sr
        return {'status': status, 'original_tuning': float(ot), 'target_tuning': float(target_tuning), 'confidence': float(d['confidence']), 'detection_method': d['method'], 'semitones_shifted': float(round(ss, 4)), 'duration_preserved': bool(abs(hd - od) < 0.05), 'original_duration': float(od), 'healed_duration': float(hd), 'already_at_target': bool(already)}
