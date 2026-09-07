"""
Music structure graph builder module.
Constructs:
1. Segment graphs: nodes = time segments, edges = temporal adjacency + cosine similarity of features > tau.
2. Chord-transition graphs: nodes = chord classes, edges = transitions weighted by count.
Supports PyTorch Geometric Data format and JSON export.
"""

from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np

# PyTorch and PyG compatibility
try:
    import torch
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    try:
        import torch
        HAS_PYG = False
    except ImportError:
        torch = None
        HAS_PYG = False


class GraphDataFallback:
    """Fallback class replicating torch_geometric.data.Data attributes if PyG is absent."""
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, **kwargs):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x.tolist() if hasattr(self.x, "tolist") else self.x,
            "edge_index": self.edge_index.tolist() if hasattr(self.edge_index, "tolist") else self.edge_index,
            "edge_attr": self.edge_attr.tolist() if hasattr(self.edge_attr, "tolist") else self.edge_attr,
        }


class MusicGraphBuilder:
    """Constructs segment and chord-transition graphs from musical audio features."""

    # Major and minor triad templates for 12 chromatic pitches
    CHORD_NAMES = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        "Cm", "C#m", "Dm", "D#m", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "A#m", "Bm",
    ]

    def __init__(self, similarity_threshold_tau: float = 0.65):
        self.tau = similarity_threshold_tau
        self.chord_templates = self._generate_chord_templates()

    def _generate_chord_templates(self) -> np.ndarray:
        """Generates 24 binary chroma templates (12 major + 12 minor triads)."""
        templates = np.zeros((24, 12), dtype=np.float32)
        major_intervals = [0, 4, 7] # root, major third, perfect fifth
        minor_intervals = [0, 3, 7] # root, minor third, perfect fifth

        for root in range(12):
            # Major
            for interval in major_intervals:
                templates[root, (root + interval) % 12] = 1.0
            templates[root] /= np.linalg.norm(templates[root]) + 1e-6

            # Minor
            for interval in minor_intervals:
                templates[root + 12, (root + interval) % 12] = 1.0
            templates[root + 12] /= np.linalg.norm(templates[root + 12]) + 1e-6

        return templates

    def build_segment_graph(
        self,
        segment_features: np.ndarray,
        track_id: Optional[str] = None,
        labels: Optional[np.ndarray] = None,
    ):
        """
        Builds a segment graph:
        - Nodes: time segments with feature vectors (shape: [num_nodes, in_channels])
        - Edges:
          1. Temporal adjacency (bidirectional between adjacent segments i <-> i+1)
          2. Semantic similarity: cosine similarity between segment features > tau
          3. Self-loops (i -> i)
        """
        num_nodes = len(segment_features)
        if num_nodes == 0:
            raise ValueError("Segment features cannot be empty.")

        # Compute cosine similarity matrix
        norms = np.linalg.norm(segment_features, axis=1, keepdims=True) + 1e-8
        norm_features = segment_features / norms
        sim_matrix = np.dot(norm_features, norm_features.T)

        src_nodes: List[int] = []
        dst_nodes: List[int] = []
        edge_weights: List[float] = []

        # 1. Self-loops
        for i in range(num_nodes):
            src_nodes.append(i)
            dst_nodes.append(i)
            edge_weights.append(1.0)

        # 2. Temporal adjacency
        for i in range(num_nodes - 1):
            # i -> i+1
            src_nodes.append(i)
            dst_nodes.append(i + 1)
            edge_weights.append(float(sim_matrix[i, i + 1]))
            # i+1 -> i
            src_nodes.append(i + 1)
            dst_nodes.append(i)
            edge_weights.append(float(sim_matrix[i + 1, i]))

        # 3. Similarity threshold edges (> tau)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and abs(i - j) > 1: # non-adjacent
                    sim = float(sim_matrix[i, j])
                    if sim >= self.tau:
                        src_nodes.append(i)
                        dst_nodes.append(j)
                        edge_weights.append(sim)

        edge_index_np = np.array([src_nodes, dst_nodes], dtype=np.int64)
        edge_attr_np = np.array(edge_weights, dtype=np.float32).reshape(-1, 1)

        if torch is not None:
            x_t = torch.tensor(segment_features, dtype=torch.float32)
            edge_index_t = torch.tensor(edge_index_np, dtype=torch.long)
            edge_attr_t = torch.tensor(edge_attr_np, dtype=torch.float32)
            y_t = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

            if HAS_PYG:
                data = Data(x=x_t, edge_index=edge_index_t, edge_attr=edge_attr_t, y=y_t)
            else:
                data = GraphDataFallback(x=x_t, edge_index=edge_index_t, edge_attr=edge_attr_t, y=y_t)
        else:
            data = GraphDataFallback(
                x=segment_features, edge_index=edge_index_np, edge_attr=edge_attr_np, y=labels
            )

        if track_id is not None:
            data.track_id = track_id

        return data

    def build_chord_transition_graph(
        self,
        chroma_frames: np.ndarray,
        track_id: Optional[str] = None,
        labels: Optional[np.ndarray] = None,
    ):
        """
        Builds a chord-transition graph:
        - Estimates chord sequence by matching chroma frames to triad templates.
        - Nodes: 24 chord classes (or active subset).
        - Edges: transitions between consecutive chords weighted by count.
        """
        # Match each chroma frame to best triad template
        norms = np.linalg.norm(chroma_frames, axis=0, keepdims=True) + 1e-8
        norm_chroma = chroma_frames / norms
        # Template shape: (24, 12), norm_chroma: (12, frames)
        correlations = np.dot(self.chord_templates, norm_chroma) # (24, frames)
        best_chords = np.argmax(correlations, axis=0) # sequence of chord indices

        transition_counts = np.zeros((24, 24), dtype=np.float32)
        for t in range(len(best_chords) - 1):
            c1 = best_chords[t]
            c2 = best_chords[t + 1]
            transition_counts[c1, c2] += 1.0

        # Create graph structure
        src_nodes: List[int] = []
        dst_nodes: List[int] = []
        edge_weights: List[float] = []

        total_transitions = np.sum(transition_counts) + 1e-6
        for i in range(24):
            for j in range(24):
                if transition_counts[i, j] > 0:
                    src_nodes.append(i)
                    dst_nodes.append(j)
                    edge_weights.append(float(transition_counts[i, j] / total_transitions))

        # Node features: 12-dim chord template representation + 12-dim frequency stats
        chord_freqs = np.bincount(best_chords, minlength=24).astype(np.float32)
        chord_freqs /= (len(best_chords) + 1e-6)
        node_features = np.hstack([self.chord_templates, chord_freqs[:, None]]) # (24, 13)

        edge_index_np = np.array([src_nodes, dst_nodes], dtype=np.int64) if src_nodes else np.zeros((2, 0), dtype=np.int64)
        edge_attr_np = np.array(edge_weights, dtype=np.float32).reshape(-1, 1) if edge_weights else np.zeros((0, 1), dtype=np.float32)

        if torch is not None:
            x_t = torch.tensor(node_features, dtype=torch.float32)
            edge_index_t = torch.tensor(edge_index_np, dtype=torch.long)
            edge_attr_t = torch.tensor(edge_attr_np, dtype=torch.float32)
            y_t = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

            if HAS_PYG:
                data = Data(x=x_t, edge_index=edge_index_t, edge_attr=edge_attr_t, y=y_t)
            else:
                data = GraphDataFallback(x=x_t, edge_index=edge_index_t, edge_attr=edge_attr_t, y=y_t)
        else:
            data = GraphDataFallback(
                x=node_features, edge_index=edge_index_np, edge_attr=edge_attr_np, y=labels
            )

        if track_id is not None:
            data.track_id = track_id

        return data

    @staticmethod
    def save_graph_to_json(graph_data, output_json_path: str):
        """Serializes graph to human-readable JSON format."""
        def serialize_item(val):
            if hasattr(val, "tolist"):
                return val.tolist()
            if isinstance(val, (int, float, str, bool)):
                return val
            if isinstance(val, list):
                return [serialize_item(x) for x in val]
            return str(val)

        d = {
            "num_nodes": int(graph_data.x.shape[0]) if hasattr(graph_data, "x") and graph_data.x is not None else 0,
            "num_edges": int(graph_data.edge_index.shape[1]) if hasattr(graph_data, "edge_index") and graph_data.edge_index is not None else 0,
            "x": serialize_item(graph_data.x),
            "edge_index": serialize_item(graph_data.edge_index),
            "edge_attr": serialize_item(graph_data.edge_attr) if hasattr(graph_data, "edge_attr") else [],
            "track_id": getattr(graph_data, "track_id", "unknown"),
        }
        with open(output_json_path, "w") as f:
            json.dump(d, f, indent=2)

    @staticmethod
    def save_graph_to_pt(graph_data, output_pt_path: str):
        """Saves graph in PyG .pt format."""
        if torch is not None:
            torch.save(graph_data, output_pt_path)
        else:
            raise RuntimeError("PyTorch is required to save .pt files.")
