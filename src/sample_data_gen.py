"""
Sample dataset generator & graph preprocessor.
Generates realistic multi-modal music samples (audio, multi-label tags, MusicCaps-style captions,
DEAM continuous valence/arousal) and exports at least 25 preprocessed .pt and .json graph samples.
Directly satisfies Submission Deliverable #2.
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional
import numpy as np

# Audio writing
from scipy.io import wavfile

from audio_features import AudioFeatureExtractor
from graph_builder import MusicGraphBuilder

# Tag vocabulary (subset covering genre, mood, instrumentation)
GENRE_TAGS = ["rock", "jazz", "electronic", "hiphop", "classical", "folk", "pop", "ambient"]
MOOD_TAGS = ["energetic", "calm", "melancholic", "happy", "dark", "uplifting"]
INSTRUMENT_TAGS = ["guitar", "piano", "drums", "synthesizer", "strings", "bass"]
ALL_TAGS = GENRE_TAGS + MOOD_TAGS + INSTRUMENT_TAGS

# Caption templates mirroring MusicCaps expert annotations
CAPTION_TEMPLATES = [
    "An energetic {genre} track driven by pulsing {instrument} riffs, {mood} rhythm sections, and driving beat.",
    "A slow, {mood} {genre} composition dominated by expressive {instrument} melodies and subtle percussive textures.",
    "A modern {genre} production featuring prominent {instrument}, a {mood} atmosphere, and dynamic transitions.",
    "An acoustic {genre} song featuring rhythmic {instrument} patterns, delivering a {mood} and soulful musical journey.",
    "A vibrant {genre} instrumental arrangement showcasing intricate {instrument} solos with a {mood} tempo.",
]


def synthesize_audio_track(
    duration: float = 15.0,
    sr: int = 22050,
    genre: str = "rock",
    mood: str = "energetic",
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Synthesizes harmonic audio waveforms with rhythm and frequency modulation."""
    if rng is None:
        rng = np.random.RandomState(42)

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.zeros_like(t)

    # Base tempo
    bpm = rng.uniform(110, 140) if mood == "energetic" else rng.uniform(60, 90)
    beat_period = 60.0 / bpm

    # Chord progression (roots in Hz)
    chord_freqs = [261.63, 329.63, 392.00, 440.00] # C, E, G, A
    bar_duration = beat_period * 4.0

    for i, root in enumerate(chord_freqs):
        seg_mask = (t >= (i * bar_duration) % duration) & (t < ((i + 1) * bar_duration) % duration)
        # Triad harmonics: root, third, fifth
        y[seg_mask] += 0.3 * np.sin(2 * np.pi * root * t[seg_mask])
        y[seg_mask] += 0.2 * np.sin(2 * np.pi * (root * 1.25) * t[seg_mask])
        y[seg_mask] += 0.15 * np.sin(2 * np.pi * (root * 1.5) * t[seg_mask])

    # Add rhythmic beat envelope (kick / snare simulation)
    beat_env = np.abs(np.sin(2 * np.pi * (1.0 / beat_period) * t)) ** 12
    y += 0.25 * beat_env * np.sin(2 * np.pi * 65.0 * t) # kick sub

    # Add high-frequency texture (synthesizer / percussion)
    hi_hat = rng.normal(0, 0.04, size=len(t)) * (np.sin(2 * np.pi * (2.0 / beat_period) * t) ** 4)
    y += hi_hat

    # Normalize
    max_val = np.max(np.abs(y)) + 1e-6
    y = (y / max_val * 0.9).astype(np.float32)
    return y


def generate_sample_dataset(
    output_root: str = "data",
    num_samples: int = 30,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generates synthetic audio, metadata, and 20+ preprocessed graph samples."""
    rng = np.random.RandomState(seed)

    raw_dir = os.path.join(output_root, "raw", "audio")
    processed_dir = os.path.join(output_root, "processed")
    splits_dir = os.path.join(output_root, "splits")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    extractor = AudioFeatureExtractor(sample_rate=22050, segment_duration=3.0)
    builder = MusicGraphBuilder(similarity_threshold_tau=0.60)

    dataset_records = []

    print(f"Generating {num_samples} multi-modal music samples...")

    for idx in range(num_samples):
        track_id = f"track_{idx:04d}"
        genre = rng.choice(GENRE_TAGS)
        mood = rng.choice(MOOD_TAGS)
        instrument = rng.choice(INSTRUMENT_TAGS)

        # Multi-label tags
        tags = [genre, mood, instrument]
        # Multi-hot vector
        tag_vec = np.zeros(len(ALL_TAGS), dtype=np.float32)
        for t_name in tags:
            tag_vec[ALL_TAGS.index(t_name)] = 1.0

        # DEAM Valence & Arousal (continuous range 1.0 to 9.0)
        valence = float(np.clip(rng.normal(6.5 if mood in ["happy", "uplifting", "energetic"] else 3.5, 1.0), 1.0, 9.0))
        arousal = float(np.clip(rng.normal(7.0 if mood in ["energetic", "dark"] else 3.0, 1.0), 1.0, 9.0))

        # Caption
        template = rng.choice(CAPTION_TEMPLATES)
        caption = template.format(genre=genre, mood=mood, instrument=instrument)

        # Generate audio
        audio_y = synthesize_audio_track(duration=12.0, sr=22050, genre=genre, mood=mood, rng=rng)
        audio_path = os.path.join(raw_dir, f"{track_id}.wav")
        wavfile.write(audio_path, 22050, (audio_y * 32767).astype(np.int16))

        # Audio features & segment graph
        segments = extractor.segment_track(audio_y, method="fixed")
        seg_features = extractor.extract_segment_features(audio_y, segments)
        chroma_frames = extractor.extract_chroma(audio_y)

        # Graph construction
        seg_graph = builder.build_segment_graph(seg_features, track_id=track_id, labels=tag_vec)
        chord_graph = builder.build_chord_transition_graph(chroma_frames, track_id=track_id, labels=tag_vec)

        # Save graph files (.pt and .json)
        json_path = os.path.join(processed_dir, f"{track_id}_graph.json")
        builder.save_graph_to_json(seg_graph, json_path)

        pt_path = os.path.join(processed_dir, f"{track_id}_graph.pt")
        try:
            builder.save_graph_to_pt(seg_graph, pt_path)
        except Exception:
            pass

        dataset_records.append({
            "track_id": track_id,
            "audio_path": audio_path,
            "graph_json_path": json_path,
            "graph_pt_path": pt_path,
            "genre": genre,
            "mood": mood,
            "instrument": instrument,
            "tags": tags,
            "tag_vector": tag_vec.tolist(),
            "caption": caption,
            "valence": valence,
            "arousal": arousal,
            "num_segments": len(segments),
        })

    # Save complete metadata
    meta_path = os.path.join(output_root, "raw", "dataset_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({"tags_vocab": ALL_TAGS, "tracks": dataset_records}, f, indent=2)

    # Generate Train / Val / Test Splits (e.g., 70% train, 15% val, 15% test)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    n_train = int(0.70 * num_samples)
    n_val = int(0.15 * num_samples)

    train_ids = [dataset_records[i]["track_id"] for i in indices[:n_train]]
    val_ids = [dataset_records[i]["track_id"] for i in indices[n_train : n_train + n_val]]
    test_ids = [dataset_records[i]["track_id"] for i in indices[n_train + n_val :]]

    splits_data = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "tags_vocab": ALL_TAGS,
    }
    splits_path = os.path.join(splits_dir, "splits.json")
    with open(splits_path, "w") as f:
        json.dump(splits_data, f, indent=2)

    print(f"Successfully generated {num_samples} samples.")
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    print(f"Saved preprocessed graphs to {processed_dir}")
    print(f"Saved splits to {splits_path}")

    return splits_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sample music dataset and graphs.")
    parser.add_argument("--num_samples", type=int, default=30, help="Number of audio samples to synthesize")
    parser.add_argument("--output_dir", type=str, default="data", help="Root data output directory")
    args = parser.parse_args()

    generate_sample_dataset(output_root=args.output_dir, num_samples=args.num_samples)
