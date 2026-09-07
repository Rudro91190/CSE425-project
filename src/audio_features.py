"""
Audio feature extraction module.
Resamples to 22,050 Hz, extracts 128-bin log-mel spectrograms and 12-bin chroma features,
normalizes per track, and handles segmentations (fixed windows or beat-synchronous).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


class AudioFeatureExtractor:
    """Extracts spectrogram, chroma, and segment-level features from audio signals."""

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 128,
        n_chroma: int = 12,
        n_fft: int = 2048,
        hop_length: int = 512,
        segment_duration: float = 5.0,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_chroma = n_chroma
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.segment_duration = segment_duration

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Loads and resamples audio to target sample rate (22,050 Hz)."""
        if HAS_LIBROSA:
            y, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            return y, sr
        else:
            # Fallback simple reader using scipy.io.wavfile or dummy
            from scipy.io import wavfile
            sr, y = wavfile.read(file_path)
            if y.ndim > 1:
                y = y.mean(axis=1)
            y = y.astype(np.float32)
            if np.max(np.abs(y)) > 0:
                y /= np.max(np.abs(y))
            return y, sr

    def extract_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """
        Extracts 128-bin log-mel spectrogram normalized per track.
        Output shape: (n_mels, frames)
        """
        if HAS_LIBROSA:
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            )
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        else:
            # Simple FFT-based filterbank approximation fallback
            n_frames = max(1, int(len(y) // self.hop_length))
            log_mel = np.zeros((self.n_mels, n_frames), dtype=np.float32)
            for i in range(min(n_frames, 200)):
                start = i * self.hop_length
                frame = y[start : start + self.n_fft]
                if len(frame) < self.n_fft:
                    frame = np.pad(frame, (0, self.n_fft - len(frame)))
                fft_mag = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
                # Pool into n_mels bins
                bin_size = max(1, len(fft_mag) // self.n_mels)
                for m in range(self.n_mels):
                    log_mel[m, i] = np.mean(fft_mag[m * bin_size : (m + 1) * bin_size] ** 2 + 1e-6)
            log_mel = 10.0 * np.log10(np.maximum(log_mel, 1e-6))

        # Per-track normalization (zero mean, unit variance)
        mean = np.mean(log_mel)
        std = np.std(log_mel) + 1e-6
        norm_mel = (log_mel - mean) / std
        return norm_mel.astype(np.float32)

    def extract_chroma(self, y: np.ndarray) -> np.ndarray:
        """
        Extracts 12-bin chroma features (pitch classes C, C#, D, ..., B).
        Output shape: (12, frames)
        """
        if HAS_LIBROSA:
            chroma = librosa.feature.chroma_stft(
                y=y,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_chroma=self.n_chroma,
            )
        else:
            # Fallback 12-bin periodic pitch pooling
            n_frames = max(1, int(len(y) // self.hop_length))
            chroma = np.zeros((self.n_chroma, n_frames), dtype=np.float32)
            for i in range(min(n_frames, 200)):
                start = i * self.hop_length
                frame = y[start : start + self.n_fft]
                if len(frame) < self.n_fft:
                    frame = np.pad(frame, (0, self.n_fft - len(frame)))
                fft_mag = np.abs(np.fft.rfft(frame))
                for c in range(self.n_chroma):
                    chroma[c, i] = np.sum(fft_mag[c::self.n_chroma])
            norm = np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-6
            chroma = chroma / norm

        return chroma.astype(np.float32)

    def segment_track(
        self, y: np.ndarray, method: str = "fixed"
    ) -> List[Tuple[int, int]]:
        """
        Splits track into fixed windows (5-10s) or beat-synchronous segments.
        Returns a list of (start_sample, end_sample) indices.
        """
        total_samples = len(y)
        window_samples = int(self.segment_duration * self.sample_rate)

        if method == "beat" and HAS_LIBROSA:
            try:
                _, beat_frames = librosa.beat.beat_track(
                    y=y, sr=self.sample_rate, hop_length=self.hop_length
                )
                beat_samples = librosa.frames_to_samples(beat_frames, hop_length=self.hop_length)
                if len(beat_samples) > 2:
                    segments = []
                    # Group beats into bars / segments of ~4 beats
                    bar_step = 4
                    for i in range(0, len(beat_samples) - 1, bar_step):
                        start = beat_samples[i]
                        end = beat_samples[min(i + bar_step, len(beat_samples) - 1)]
                        if end > start:
                            segments.append((int(start), int(end)))
                    if segments:
                        return segments
            except Exception:
                pass

        # Default: fixed-duration windows
        segments = []
        for start in range(0, total_samples, window_samples):
            end = min(start + window_samples, total_samples)
            if end - start >= self.sample_rate: # at least 1s
                segments.append((start, end))
        if not segments:
            segments.append((0, total_samples))
        return segments

    def extract_segment_features(
        self, y: np.ndarray, segments: List[Tuple[int, int]]
    ) -> np.ndarray:
        """
        Extracts pooled (mel 128 + chroma 12 = 140 dim) feature vectors for each segment.
        Output shape: (num_segments, 140)
        """
        num_segments = len(segments)
        features = np.zeros((num_segments, self.n_mels + self.n_chroma), dtype=np.float32)

        for idx, (start, end) in enumerate(segments):
            seg_y = y[start:end]
            if len(seg_y) < self.hop_length * 2:
                seg_y = np.pad(seg_y, (0, self.hop_length * 2 - len(seg_y)))

            mel = self.extract_mel_spectrogram(seg_y)
            chroma = self.extract_chroma(seg_y)

            # Mean-pool along time dimension
            mel_mean = np.mean(mel, axis=1) # 128
            chroma_mean = np.mean(chroma, axis=1) # 12

            features[idx, : self.n_mels] = mel_mean
            features[idx, self.n_mels :] = chroma_mean

        return features
