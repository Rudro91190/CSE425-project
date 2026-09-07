"""
Dataset and DataLoader module for multi-modal music understanding.
Loads audio spectrograms, graph structures, text tokens, and multi-task labels.
"""

import os
import json
from typing import Dict, List, Optional, Any
import numpy as np

import torch
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from audio_features import AudioFeatureExtractor
from graph_builder import MusicGraphBuilder


class MusicMultimodalDataset(Dataset):
    """
    PyTorch Dataset providing:
    - Text inputs: tokenized tags, captions, or lyrics
    - Audio inputs: 128-bin log-mel spectrogram for CNN baseline
    - Graph inputs: PyG Data object (segment graph)
    - Labels: multi-label tag vector, continuous valence and arousal values
    """

    def __init__(
        self,
        metadata_file: str,
        split_file: str,
        split: str = "train",
        tokenizer=None,
        max_text_length: int = 128,
        use_captions_for_text: bool = True,
    ):
        self.split = split
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.use_captions_for_text = use_captions_for_text

        with open(metadata_file, "r") as f:
            meta_data = json.load(f)
            self.tracks_dict = {t["track_id"]: t for t in meta_data["tracks"]}
            self.tags_vocab = meta_data["tags_vocab"]

        with open(split_file, "r") as f:
            splits_data = json.load(f)
            self.track_ids = splits_data[split]

        self.extractor = AudioFeatureExtractor()
        self.builder = MusicGraphBuilder()

    def __len__(self) -> int:
        return len(self.track_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        track_id = self.track_ids[idx]
        record = self.tracks_dict[track_id]

        # 1. Text input
        text_str = record["caption"] if self.use_captions_for_text else " ".join(record["tags"])
        text_data = {"text": text_str}

        if self.tokenizer is not None:
            enc = self.tokenizer(
                text_str,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            )
            text_data["input_ids"] = enc["input_ids"].squeeze(0)
            text_data["attention_mask"] = enc["attention_mask"].squeeze(0)
        else:
            # Fallback simple bag-of-words or ascii dummy tokenization
            tokens = [min(255, ord(c)) for c in text_str[: self.max_text_length]]
            if len(tokens) < self.max_text_length:
                mask = [1] * len(tokens) + [0] * (self.max_text_length - len(tokens))
                tokens = tokens + [0] * (self.max_text_length - len(tokens))
            else:
                mask = [1] * self.max_text_length
            text_data["input_ids"] = torch.tensor(tokens, dtype=torch.long)
            text_data["attention_mask"] = torch.tensor(mask, dtype=torch.long)

        # 2. Graph structure
        # Check if preprocessed .pt or .json exists
        graph_pt = record.get("graph_pt_path", "")
        graph_json = record.get("graph_json_path", "")

        graph_data = None
        if os.path.exists(graph_pt):
            try:
                graph_data = torch.load(graph_pt)
            except Exception:
                graph_data = None

        if graph_data is None and os.path.exists(graph_json):
            with open(graph_json, "r") as f:
                gj = json.load(f)
                x = torch.tensor(gj["x"], dtype=torch.float32)
                edge_index = torch.tensor(gj["edge_index"], dtype=torch.long)
                edge_attr = torch.tensor(gj["edge_attr"], dtype=torch.float32)
                if HAS_PYG:
                    from torch_geometric.data import Data
                    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                else:
                    from graph_builder import GraphDataFallback
                    graph_data = GraphDataFallback(x=x, edge_index=edge_index, edge_attr=edge_attr)

        if graph_data is None:
            # Recompute on the fly
            audio_y, sr = self.extractor.load_audio(record["audio_path"])
            segs = self.extractor.segment_track(audio_y)
            seg_feats = self.extractor.extract_segment_features(audio_y, segs)
            graph_data = self.builder.build_segment_graph(seg_feats, track_id=track_id)

        # 3. Audio Spectrogram for Baseline CNN (fixed size: 128 x 256)
        audio_y, sr = self.extractor.load_audio(record["audio_path"])
        mel = self.extractor.extract_mel_spectrogram(audio_y)
        target_frames = 256
        if mel.shape[1] < target_frames:
            mel = np.pad(mel, ((0, 0), (0, target_frames - mel.shape[1])), mode="constant")
        else:
            mel = mel[:, :target_frames]
        spectrogram_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0) # (1, 128, 256)

        # 4. Labels
        tag_vec = torch.tensor(record["tag_vector"], dtype=torch.float32)
        valence = torch.tensor(record.get("valence", 5.0), dtype=torch.float32)
        arousal = torch.tensor(record.get("arousal", 5.0), dtype=torch.float32)

        return {
            "track_id": track_id,
            "text_data": text_data,
            "graph": graph_data,
            "spectrogram": spectrogram_tensor,
            "tag_vec": tag_vec,
            "valence": valence,
            "arousal": arousal,
            "caption": record["caption"],
        }


def collate_multimodal_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collator handling graphs, text sequences, and spectrogram batches."""
    track_ids = [item["track_id"] for item in batch]
    captions = [item["caption"] for item in batch]

    input_ids = torch.stack([item["text_data"]["input_ids"] for item in batch])
    attention_mask = torch.stack([item["text_data"]["attention_mask"] for item in batch])
    spectrograms = torch.stack([item["spectrogram"] for item in batch])
    tag_vecs = torch.stack([item["tag_vec"] for item in batch])
    valences = torch.stack([item["valence"] for item in batch])
    arousals = torch.stack([item["arousal"] for item in batch])

    # PyG Batching
    graphs = [item["graph"] for item in batch]
    if HAS_PYG:
        batched_graph = Batch.from_data_list(graphs)
    else:
        # Simple manual batch representation
        batched_graph = graphs

    return {
        "track_ids": track_ids,
        "captions": captions,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "spectrograms": spectrograms,
        "graphs": batched_graph,
        "tag_vecs": tag_vecs,
        "valences": valences,
        "arousals": arousals,
    }
