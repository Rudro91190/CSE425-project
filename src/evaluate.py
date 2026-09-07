"""
Evaluation and Metrics module.
Computes:
- Tag/genre classification: Macro-F1, Micro-F1, per-tag AUC-PR, mean AUC-PR
- Emotion regression: MAE, R^2 for valence and arousal
- t-SNE visualization of fusion vectors z colored by genre and mood
- Case studies analyzing graph paths + cross-attention alignment
- Formatted metrics export to results/metrics.json
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

try:
    from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def compute_classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes Macro-F1, Micro-F1, and mean AUC-PR for multi-label classification.
    y_true: [N, K] binary {0, 1}
    y_prob: [N, K] continuous probabilities in [0, 1]
    """
    y_pred = (y_prob >= threshold).astype(np.float32)
    num_classes = y_true.shape[1]

    f1_per_class = []
    auc_pr_per_class = []

    # Micro counts
    tp_total = np.sum((y_true == 1) & (y_pred == 1))
    fp_total = np.sum((y_true == 0) & (y_pred == 1))
    fn_total = np.sum((y_true == 1) & (y_pred == 0))

    micro_prec = tp_total / (tp_total + fp_total + 1e-8)
    micro_rec = tp_total / (tp_total + fn_total + 1e-8)
    micro_f1 = 2 * (micro_prec * micro_rec) / (micro_prec + micro_rec + 1e-8)

    for k in range(num_classes):
        yt = y_true[:, k]
        yp = y_pred[:, k]
        probs = y_prob[:, k]

        tp = np.sum((yt == 1) & (yp == 1))
        fp = np.sum((yt == 0) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))

        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
        f1_per_class.append(float(f1))

        if HAS_SKLEARN and np.sum(yt) > 0 and np.sum(yt) < len(yt):
            try:
                p_curve, r_curve, _ = precision_recall_curve(yt, probs)
                class_auc = auc(r_curve, p_curve)
            except Exception:
                class_auc = float(prec)
        else:
            class_auc = float(prec)
        auc_pr_per_class.append(float(class_auc))

    macro_f1 = float(np.mean(f1_per_class))
    mean_auc_pr = float(np.mean(auc_pr_per_class))

    return {
        "macro_f1": macro_f1,
        "micro_f1": float(micro_f1),
        "mean_auc_pr": mean_auc_pr,
        "f1_per_class": f1_per_class,
        "auc_pr_per_class": auc_pr_per_class,
    }


def compute_emotion_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Computes MAE and R^2 score for continuous emotion regression."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    y_bar = np.mean(y_true)
    ss_tot = np.sum((y_true - y_bar) ** 2) + 1e-8
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot))
    return {"mae": mae, "r2": r2}


def plot_tsne_embeddings(
    z_embeddings: np.ndarray,
    labels_list: List[str],
    title: str = "t-SNE of GNN-BERT Fusion Embeddings (z)",
    output_png_path: str = "results/plots/tsne_fusion_embeddings.png",
):
    """Computes t-SNE 2D projection and saves plot colored by genre/mood."""
    if not HAS_SKLEARN:
        print("Scikit-learn not available for t-SNE.")
        return

    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    n_samples = len(z_embeddings)
    if n_samples < 3:
        print("Too few samples for t-SNE projection.")
        return
    perplexity = min(30.0, max(1.0, float(min(10.0, n_samples - 1))))

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    z_2d = tsne.fit_transform(z_embeddings)

    unique_labels = sorted(list(set(labels_list)))
    color_map = plt.get_cmap("tab10")

    plt.figure(figsize=(9, 7))
    for idx, label in enumerate(unique_labels):
        indices = [i for i, l in enumerate(labels_list) if l == label]
        plt.scatter(
            z_2d[indices, 0],
            z_2d[indices, 1],
            color=color_map(idx),
            label=label,
            alpha=0.8,
            edgecolors="none",
            s=40,
        )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("t-SNE Dimension 1", fontsize=11)
    plt.ylabel("t-SNE Dimension 2", fontsize=11)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Category")
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300)
    plt.close()
    print(f"Saved t-SNE plot to {output_png_path}")


def generate_case_studies(
    model,
    dataset,
    device: str = "cpu",
    output_json_path: str = "results/case_studies.json",
    num_cases: int = 3,
):
    """
    Generates 3 in-depth case studies showing graph paths,
    cross-attention alignment weights, and prediction outputs.
    """
    model.eval()
    cases = []

    for i in range(min(num_cases, len(dataset))):
        sample = dataset[i]
        track_id = sample["track_id"]
        caption = sample["caption"]
        ground_truth_tags = [
            dataset.tags_vocab[k] for k, v in enumerate(sample["tag_vec"]) if v == 1.0
        ]

        # Graph node info
        graph = sample["graph"]
        num_nodes = int(graph.x.shape[0]) if hasattr(graph, "x") else len(graph["x"])
        num_edges = int(graph.edge_index.shape[1]) if hasattr(graph, "edge_index") else len(graph["edge_index"][0])

        input_ids = sample["text_data"]["input_ids"].unsqueeze(0).to(device)
        attn_mask = sample["text_data"]["attention_mask"].unsqueeze(0).to(device)

        with torch.no_grad():
            x = graph.x.to(device)
            edge_index = graph.edge_index.to(device)
            out = model(
                x=x,
                edge_index=edge_index,
                batch=torch.zeros(x.size(0), dtype=torch.long, device=device),
                input_ids=input_ids,
                attention_mask=attn_mask,
            )

        probs = out["probs"].squeeze(0).cpu().numpy()
        pred_top_tags = [
            (dataset.tags_vocab[idx], float(probs[idx]))
            for idx in np.argsort(-probs)[:5]
        ]

        # Cross-attention weights over caption tokens
        attn_weights = out.get("attn_weights")
        if attn_weights is not None:
            top_attn_tokens = float(np.max(attn_weights.cpu().numpy()))
        else:
            top_attn_tokens = None

        cases.append({
            "case_id": i + 1,
            "track_id": track_id,
            "caption": caption,
            "ground_truth_tags": ground_truth_tags,
            "predicted_top_tags": pred_top_tags,
            "graph_structure": {
                "num_temporal_nodes": num_nodes,
                "num_structural_edges": num_edges,
                "segment_duration_seconds": 3.0,
            },
            "emotion_predictions": {
                "pred_valence": float(out["pred_valence"].item()),
                "ground_truth_valence": float(sample["valence"].item()),
                "pred_arousal": float(out["pred_arousal"].item()),
                "ground_truth_arousal": float(sample["arousal"].item()),
            },
            "cross_modal_alignment_score": top_attn_tokens,
        })

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"Saved {len(cases)} case studies to {output_json_path}")
    return cases
