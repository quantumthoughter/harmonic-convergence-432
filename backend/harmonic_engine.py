"""
Harmonic Engine v3 — ambient Solfeggio, crystal binaural, Schumann stack, Heart, Singularity, Infrasound.
"""
import numpy as np

class HarmonicEngine:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.solfeggio_freqs = {
            '174 Hz (Pain Relief)': 174.0, '285 Hz (Healing)': 285.0, '396 Hz (Liberation)': 396.0,
            '417 Hz (Change)': 417.0, '528 Hz (DNA Repair)': 528.0, '639 Hz (Connection)': 639.0,
            '741 Hz (Expression)': 741.0, '852 Hz (Intuition)': 852.0, '963 Hz (Transcendence)': 963.0}
        self.schumann_modes = {
            '7.83 Hz (Fundamental)': 7.83, '14.3 Hz (2nd)': 14.3, '20.8 Hz (3rd)': 20.8,
            '27.3 Hz (4th)': 27.3, '33.8 Hz (5th)': 33.8, '39.0 Hz (6th)': 39.0}

    def generate_solfeggio(self, freq, duration, amplitude=0.3, sr=None):
        if sr is None: sr = self.sample_rate
        n = int(sr * duration); t = np.linspace(0, duration, n, endpoint=False)
        env = np.ones(n); atk = min(int(0.15 * sr), n); rel = min(int(0.8 * sr), n)
        env[:atk] = np.linspace(0, 1, atk); env[-rel:] = np.linspace(1, 0, rel)
        lfo = 1.0 + 0.4 * np.sin(2 * np.pi * 0.15 * t)
        tone = amplitude * np.sin(2 * np.pi * freq * t) * env * lfo
        ot = amplitude * 0.25 * np.sin(2 * np.pi * freq * 3 * t) * env * lfo
        stereo = np.zeros((2, n), dtype=np.float64)
        stereo[0] = tone * 0.7 + ot * 0.3; stereo[1] = tone * 0.5 + ot * 0.7
        return stereo

    def mix_all(self, healed_audio, sr, config):
        dur = healed_audio.shape[-1] / sr
        if healed_audio.ndim == 1:
            mix = np.zeros((2, len(healed_audio)), dtype=np.float64); mix[0] = healed_audio; mix[1] = healed_audio
        elif healed_audio.ndim == 2 and healed_audio.shape[0] == 2:
            mix = healed_audio.copy().astype(np.float64)
        else:
            mix = np.zeros((2, len(healed_audio)), dtype=np.float64); mix[0] = healed_audio; mix[1] = healed_audio
        n = mix.shape[1]
        has = (any(v > 0.01 for v in config.get('solfeggio', {}).values()) or
               config.get('binaural', {}).get('enabled', False) or
               config.get('schumann', {}).get('enabled', False) or
               config.get('heart_coherence', {}).get('enabled', False) or
               config.get('singularity', {}).get('enabled', False) or
               config.get('infrasound', {}).get('enabled', False))
        if not has:
            peak = np.max(np.abs(mix))
            if peak > 1.0: mix = mix / peak * 0.98
            return mix.astype(np.float32)
        lvl = np.sqrt(np.mean(mix.astype(np.float64)**2)) + 1e-10
        for name, vol in config.get('solfeggio', {}).items():
            if vol > 0.01 and name in self.solfeggio_freqs:
                f = self.solfeggio_freqs[name]
                layer = self.generate_solfeggio(f, dur, amplitude=vol * 0.4 * lvl, sr=sr)
                mix[:, :n] += layer[:, :n]
        master = config.get('master_volume', 1.0)
        peak = np.max(np.abs(mix))
        if peak > 1.0: mix = mix / peak * 0.98
        if master < 1.0: mix = mix * master
        return mix.astype(np.float32)
