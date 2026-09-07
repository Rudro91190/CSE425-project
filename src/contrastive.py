"""
Cross-Modal Contrastive Alignment module (Task 4).
Learns a shared embedding space between music structure graphs and natural language captions (MusicCaps)
using InfoNCE dual-encoder training.
Implements bidirectional R@1, R@5, R@10 retrieval and zero-shot tag prediction.
"""

from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_encoder import MusicBERTClassifier
from gnn_model import MusicGNNEncoder


class DualEncoderGNNBERT(nn.Module):
    """
    Dual-Encoder architecture for cross-modal contrastive learning:
    - Audio branch: GNN encodes segment graph -> projects to shared D-dim space -> L2 normalize
    - Text branch: BERT encodes caption/lyrics -> projects to shared D-dim space -> L2 normalize
    """

    def __init__(
        self,
        embed_dim: int = 128,
        temperature: float = 0.07,
        d_gnn: int = 128,
        bert_model_name: str = "distilbert-base-uncased",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature

        # Graph encoder
        self.gnn = MusicGNNEncoder(
            in_channels=140,
            hidden_channels=d_gnn,
            out_channels=d_gnn,
            gnn_type="GraphSAGE",
        )
        self.graph_proj = nn.Sequential(
            nn.Linear(d_gnn, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Text encoder
        self.bert = MusicBERTClassifier(model_name=bert_model_name)
        self.text_proj = nn.Sequential(
            nn.Linear(self.bert.d_model, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def encode_graph(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor]) -> torch.Tensor:
        """Encodes graph structure into normalized embedding vector g_i."""
        gnn_out = self.gnn(x, edge_index, batch)
        g_raw = gnn_out["graph_embedding"]
        g_proj = self.graph_proj(g_raw)
        return F.normalize(g_proj, p=2, dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encodes text caption into normalized embedding vector t_i."""
        bert_out = self.bert(input_ids, attention_mask)
        t_raw = bert_out["cls_token"]
        t_proj = self.text_proj(t_raw)
        return F.normalize(t_proj, p=2, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes symmetric InfoNCE contrastive loss over mini-batch.
        """
        g = self.encode_graph(x, edge_index, batch) # [N, D]
        t = self.encode_text(input_ids, attention_mask) # [N, D]

        batch_size = g.size(0)

        # Cosine similarity matrix S_ij = g_i^T t_j / tau
        sim_matrix = torch.matmul(g, t.T) / self.temperature # [N, N]

        # Targets are diagonal indices
        targets = torch.arange(batch_size, device=g.device)

        # Symmetric InfoNCE loss
        loss_g2t = F.cross_entropy(sim_matrix, targets)
        loss_t2g = F.cross_entropy(sim_matrix.T, targets)
        loss = (loss_g2t + loss_t2g) / 2.0

        return {
            "loss": loss,
            "sim_matrix": sim_matrix,
            "graph_embeds": g,
            "text_embeds": t,
        }


def evaluate_cross_modal_retrieval(
    graph_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    captions: List[str],
    track_ids: List[str],
    output_qualitative_json: Optional[str] = None,
) -> Dict[str, float]:
    """
    Computes standard cross-modal retrieval metrics:
    - Caption -> Audio: R@1, R@5, R@10, Mean Rank
    - Audio -> Caption: R@1, R@5, R@10, Mean Rank
    Exports qualitative top-3 matched examples.
    """
    num_samples = len(track_ids)
    # Cosine similarity matrix
    sim = torch.matmul(text_embeds, graph_embeds.T).detach().cpu().numpy() # [N_text, N_audio]

    def compute_recall_at_k(similarity_mat: torch.Tensor) -> Dict[str, float]:
        n = similarity_mat.shape[0]
        ranks = []
        r1, r5, r10 = 0, 0, 0

        for i in range(n):
            sorted_indices = np.argsort(-similarity_mat[i])
            rank = np.where(sorted_indices == i)[0][0] + 1
            ranks.append(rank)
            if rank <= 1:
                r1 += 1
            if rank <= 5:
                r5 += 1
            if rank <= 10:
                r10 += 1

        return {
            "R@1": float(r1 / n),
            "R@5": float(r5 / n),
            "R@10": float(r10 / n),
            "MeanRank": float(np.mean(ranks)),
            "MRR": float(np.mean(1.0 / np.array(ranks))),
        }

    c2a_metrics = compute_recall_at_k(sim)
    a2c_metrics = compute_recall_at_k(sim.T)

    # 10 qualitative retrieval examples
    qualitative_results = []
    num_examples = min(10, num_samples)
    for i in range(num_examples):
        sorted_indices = np.argsort(-sim[i])
        top3_indices = sorted_indices[:3].tolist()
        qualitative_results.append({
            "query_caption": captions[i],
            "ground_truth_track": track_ids[i],
            "top_matches": [
                {
                    "rank": r + 1,
                    "matched_track": track_ids[match_idx],
                    "similarity_score": float(sim[i, match_idx]),
                    "is_correct": bool(match_idx == i),
                }
                for r, match_idx in enumerate(top3_indices)
            ],
        })

    if output_qualitative_json is not None:
        with open(output_qualitative_json, "w") as f:
            json.dump(qualitative_results, f, indent=2)

    return {
        "Caption_to_Audio_R1": c2a_metrics["R@1"],
        "Caption_to_Audio_R5": c2a_metrics["R@5"],
        "Caption_to_Audio_R10": c2a_metrics["R@10"],
        "Audio_to_Caption_R1": a2c_metrics["R@1"],
        "Audio_to_Caption_R5": a2c_metrics["R@5"],
        "Audio_to_Caption_R10": a2c_metrics["R@10"],
        "Caption_to_Audio_MRR": c2a_metrics["MRR"],
    }
