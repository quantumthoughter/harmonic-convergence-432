"""
Atlantean Kernel v4 — 8×8 multi-segment CQT detection with probability voting.
"""
import numpy as np
import librosa
import soundfile as sf
import subprocess, os, json
from datetime import datetime

class AtlanteanKernel:
    VOTE_BINS = [420, 424, 427, 430, 432, 435, 438, 440, 442, 444, 445, 446, 448]
    SEGMENTS = 8
    SEGMENT_DURATION = 8.0

    def __init__(self):
        self.target_tuning = 432.0
        self.sample_rate = 44100
        self._detect_cache = {}

    def _round_meaningful(self, raw_tuning):
        if 430.0 <= raw_tuning <= 434.0:
            return 432.0
        for target in self.VOTE_BINS:
            if abs(raw_tuning - target) <= 0.5:
                return float(target)
        return round(raw_tuning, 2)

    def _cqt_segment(self, y, sr, offset_seconds):
        start_sample = int(offset_seconds * sr)
        end_sample = start_sample + int(self.SEGMENT_DURATION * sr)
        if end_sample > len(y):
            end_sample = len(y)
            start_sample = max(0, end_sample - int(self.SEGMENT_DURATION * sr))
        if start_sample < 0 or end_sample - start_sample < sr:
            return None, 0.0
        seg = y[start_sample:end_sample]
        C = np.abs(librosa.cqt(seg, sr=sr, bins_per_octave=72, n_bins=72*7, fmin=32.7))
        freqs = librosa.cqt_frequencies(n_bins=72*7, fmin=32.7, bins_per_octave=72)
        peaks_idx = np.argsort(C.max(axis=1))[-300:]
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
        if confidence > 0.3 and (sorted_scores[0][1] + sorted_scores[1][1]) > 0:
            best = (best * sorted_scores[0][1] + second * sorted_scores[1][1]) / (sorted_scores[0][1] + sorted_scores[1][1])
        return round(best, 3), float(confidence)

    def detect_tuning_reference(self, file_path):
        y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        total_dur = len(y) / sr
        if total_dur <= self.SEGMENT_DURATION * 1.5:
            seg_len = min(int(total_dur * sr), len(y))
            y = y[:seg_len]
            best, conf = self._cqt_segment(y, sr, 0)
            if best is None:
                return (432.0, 0.0, 'cqt_grid', 432.0)
            raw = best
            if conf > 0.05:
                reported = self._round_meaningful(raw)
            else:
                reported = raw
            return (reported, float(conf), 'cqt_grid', raw)
        positions = []
        for i in range(self.SEGMENTS):
            frac = (i + 0.5) / self.SEGMENTS
            offset = frac * total_dur
            max_start = total_dur - self.SEGMENT_DURATION
            if max_start <= 0:
                offset = 0
            else:
                offset = min(offset, max_start)
            positions.append(offset)
        votes = {}
        for offset in positions:
            best, conf = self._cqt_segment(y, sr, offset)
            if best is None:
                continue
            binned = self._round_meaningful(best)
            if binned not in votes:
                votes[binned] = 0.0
            votes[binned] += conf
        if not votes:
            return (432.0, 0.0, 'cqt_grid', 432.0)
        total_weight = sum(votes.values())
        winner = max(votes, key=votes.get)
        winner_weight = votes[winner]
        final_confidence = winner_weight / total_weight if total_weight > 0 else 0.0
        second_weight = sorted(votes.values(), reverse=True)[1] if len(votes) > 1 else 0.0
        vote_confidence = max(0, min(1, (winner_weight - second_weight) / max(total_weight, 1e-10) * 2))
        raw = float(winner)
        if final_confidence > 0.05:
            reported = self._round_meaningful(raw)
        else:
            reported = raw
        return (reported, float(final_confidence), 'cqt_grid', raw)

    def detect_tuning(self, file_path, fast=False):
        if file_path in self._detect_cache:
            return self._detect_cache[file_path]
        if fast:
            y, sr = librosa.load(file_path, sr=self.sample_rate, duration=5, mono=True)
            offset = librosa.estimate_tuning(y=y, sr=sr, resolution=0.01)
            tuning = round(440.0 * (2 ** (offset / 12)), 2)
            conf = float(max(0, min(1, 1 - abs(offset) * 5)))
            result = {'tuning': tuning, 'confidence': conf, 'method': 'a4_quick', 'raw_tuning': tuning}
        else:
            tuning, conf, method, raw = self.detect_tuning_reference(file_path)
            result = {'tuning': tuning, 'confidence': conf, 'method': method, 'raw_tuning': raw}
        self._detect_cache[file_path] = result
        return result

    def shift_pitch(self, input_path, output_path, semitones):
        result = subprocess.run(["rubberband", "-t", "1.0", "-p", str(semitones), input_path, output_path], capture_output=True, text=True)
        if result.returncode != 0: raise RuntimeError(f"Rubberband failed: {result.stderr}")

    def full_heal(self, input_path, output_path, target_tuning=432.0):
        d = self.detect_tuning(input_path)
        ot = d['tuning']
        import shutil
        already = abs(ot - target_tuning) <= 0.05
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

    def verify_tuning(self, file_path, target=432.0):
        try:
            d = self.detect_tuning(file_path)
            delta = abs(d['tuning'] - target)
            return {'tuning': d['tuning'], 'delta': delta, 'pass': delta <= 1.0, 'confidence': d['confidence']}
        except Exception as e:
            return {'tuning': None, 'delta': 999, 'pass': False, 'confidence': 0, 'error': str(e)}

    def fast_detect_folder(self, folder_path):
        import glob, concurrent.futures
        AUDIO_EXTS = {'.mp3','.wav','.flac','.ogg','.m4a','.aac'}
        files = sorted([f for f in glob.glob(os.path.join(folder_path, '*')) if os.path.splitext(f)[1].lower() in AUDIO_EXTS])
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
            futs = {ex.submit(self.detect_tuning, f, True): f for f in files}
            for fut in concurrent.futures.as_completed(futs):
                f = futs[fut]
                try: results[f] = fut.result()
                except: results[f] = None
        return results

    def clear_cache(self):
        self._detect_cache.clear()
