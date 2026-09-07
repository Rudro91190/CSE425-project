"""
Unified training script for all tasks:
- Task 1: BERT Multi-Label Tag Classifier (Algorithm 1)
- Task 2: GNN on Music Segment Graphs (Algorithm 2)
- Task 3: GNN-BERT Cross-Attention Fusion & Multi-Task Loss (Algorithm 3)
- Task 4: Contrastive Dual-Encoder (Algorithm 4)
- Baseline: Mel-Spectrogram CNN (B2)
"""

import os
import sys
import json
import argparse
import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import MusicMultimodalDataset, collate_multimodal_batch
from bert_encoder import MusicBERTClassifier
from gnn_model import MusicGNNEncoder
from fusion_model import MusicGNNBERTFusionModel
from contrastive import DualEncoderGNNBERT, evaluate_cross_modal_retrieval
from baselines import MelSpectrogramCNN, BaselineRandom
from evaluate import (
    compute_classification_metrics,
    compute_emotion_metrics,
    plot_tsne_embeddings,
    generate_case_studies,
)


def load_config(config_path: str = "config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def train_task_1(args, cfg, device):
    """Task 1: BERT Multi-Label Tag Classifier."""
    print("\n--- Training Task 1: BERT Multi-Label Tag Classifier ---")
    train_ds = MusicMultimodalDataset("data/raw/dataset_metadata.json", "data/splits/splits.json", split="train")
    val_ds = MusicMultimodalDataset("data/raw/dataset_metadata.json", "data/splits/splits.json", split="val")
    test_ds = MusicMultimodalDataset("data/raw/dataset_metadata.json", "data/splits/splits.json", split="test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_multimodal_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multimodal_batch)

    num_classes = len(train_ds.tags_vocab)
    model = MusicBERTClassifier(
        model_name=cfg.get("text", {}).get("model_name", "distilbert-base-uncased"),
        num_classes=num_classes,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["tag_vecs"].to(device)

            out = model(input_ids, attn_mask, labels=labels)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {total_loss / max(1, len(train_loader)):.4f}")

    # Evaluate
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            out = model(input_ids, attn_mask)
            all_preds.append(out["probs"].cpu().numpy())
            all_trues.append(batch["tag_vecs"].numpy())

    y_prob = np.vstack(all_preds)
    y_true = np.vstack(all_trues)
    metrics = compute_classification_metrics(y_true, y_prob)
    print(f"Task 1 Test Results -> Macro-F1: {metrics['macro_f1']:.4f}, AUC-PR: {metrics['mean_auc_pr']:.4f}")
    return metrics


def train_task_2(args, cfg, device):
    """Task 2: GNN on Music Segment Graphs."""
    print(f"\n--- Training Task 2: GNN ({args.gnn_type}) on Music Structure Graphs ---")
    train_ds = MusicMultimodalDataset("data/raw/dataset_metadata.json", "data/splits/splits.json", split="train")
    test_ds = MusicMultimodalDataset("data/raw/dataset_metadata.json", "data/splits/splits.json", split="test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_multimodal_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multimodal_batch)

    num_classes = len(train_ds.tags_vocab)
    model = MusicGNNEncoder(
        in_channels=140,
        hidden_channels=128,
        out_channels=128,
        num_classes=num_classes,
        gnn_type=args.gnn_type,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            graphs = batch["graphs"]
            labels = batch["tag_vecs"].to(device)

            if hasattr(graphs, "x"):
                # PyG Batch
                out = model(graphs.x.to(device), graphs.edge_index.to(device), graphs.batch.to(device), labels=labels)
            else:
                # Iterate list of graphs
                probs_list = []
                batch_loss = 0.0
                for g_idx, g in enumerate(graphs):
                    o = model(g.x.to(device), g.edge_index.to(device), labels=labels[g_idx:g_idx+1])
                    batch_loss += o["loss"]
                loss = batch_loss / len(graphs)
                out = {"loss": loss}

            loss = out["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {total_loss / max(1, len(train_loader)):.4f}")

    # Evaluate
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            graphs = batch["graphs"]
            if hasattr(graphs, "x"):
                out = model(graphs.x.to(device), graphs.edge_index.to(device), graphs.batch.to(device))
                probs = out["probs"].cpu().numpy()
            else:
                probs = np.vstack([
                    model(g.x.to(device), g.edge_index.to(device))["probs"].cpu().numpy()
                    for g in graphs
                ])
            all_preds.append(probs)
            all_trues.append(batch["tag_vecs"].numpy())

    y_prob = np.vstack(all_preds)
    y_true = np.vstack(all_trues)
    metrics = compute_classification_metrics(y_true, y_prob)
    print(f"Task 2 Test Results -> Macro-F1: {metrics['macro_f1']:.4f}, AUC-PR: {metrics['mean_auc_pr']:.4f}")
    return metrics


def train_task_3(args, cfg, device):
    """Task 3: GNN-BERT Fusion for Multi-Context Understanding."""
    print(f"\n--- Training Task 3: GNN-BERT Fusion ({args.fusion_type}) ---")
    train_ds = MusicMultimodalDataset("data/raw/dataset_metadata.json", "data/splits/splits.json", split="train")
    test_ds = MusicMultimodalDataset("data/raw/dataset_metadata.json", "data/splits/splits.json", split="test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_multimodal_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multimodal_batch)

    num_classes = len(train_ds.tags_vocab)
    model = MusicGNNBERTFusionModel(
        num_classes=num_classes,
        fusion_type=args.fusion_type,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            graphs = batch["graphs"]
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            tag_labels = batch["tag_vecs"].to(device)
            valences = batch["valences"].to(device)
            arousals = batch["arousals"].to(device)

            if hasattr(graphs, "x"):
                out = model(
                    x=graphs.x.to(device),
                    edge_index=graphs.edge_index.to(device),
                    batch=graphs.batch.to(device),
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    tag_labels=tag_labels,
                    valence_targets=valences,
                    arousal_targets=arousals,
                )
                loss = out["loss"]
            else:
                # Iterate graphs
                losses = []
                for idx, g in enumerate(graphs):
                    o = model(
                        x=g.x.to(device),
                        edge_index=g.edge_index.to(device),
                        batch=None,
                        input_ids=input_ids[idx:idx+1],
                        attention_mask=attn_mask[idx:idx+1],
                        tag_labels=tag_labels[idx:idx+1],
                        valence_targets=valences[idx:idx+1],
                        arousal_targets=arousals[idx:idx+1],
                    )
                    losses.append(o["loss"])
                loss = torch.mean(torch.stack(losses))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{args.epochs}] Multi-Task Loss: {total_loss / max(1, len(train_loader)):.4f}")

    # Evaluate
    model.eval()
    all_preds, all_trues = [], []
    all_val_preds, all_val_trues = [], []
    all_z = []
    with torch.no_grad():
        for batch in test_loader:
            graphs = batch["graphs"]
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            if hasattr(graphs, "x"):
                out = model(
                    x=graphs.x.to(device),
                    edge_index=graphs.edge_index.to(device),
                    batch=graphs.batch.to(device),
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                )
                all_preds.append(out["probs"].cpu().numpy())
                all_val_preds.append(out["pred_valence"].cpu().numpy())
                all_z.append(out["z"].cpu().numpy())
            else:
                p_list, v_list, z_list = [], [], []
                for idx, g in enumerate(graphs):
                    o = model(
                        x=g.x.to(device),
                        edge_index=g.edge_index.to(device),
                        batch=None,
                        input_ids=input_ids[idx:idx+1],
                        attention_mask=attn_mask[idx:idx+1],
                    )
                    p_list.append(o["probs"].cpu().numpy())
                    v_list.append(o["pred_valence"].cpu().numpy())
                    z_list.append(o["z"].cpu().numpy())
                all_preds.append(np.vstack(p_list))
                all_val_preds.append(np.concatenate(v_list))
                all_z.append(np.vstack(z_list))

            all_trues.append(batch["tag_vecs"].numpy())
            all_val_trues.append(batch["valences"].numpy())

    y_prob = np.vstack(all_preds)
    y_true = np.vstack(all_trues)
    v_pred = np.concatenate(all_val_preds)
    v_true = np.concatenate(all_val_trues)

    tag_metrics = compute_classification_metrics(y_true, y_prob)
    emo_metrics = compute_emotion_metrics(v_true, v_pred)

    print(f"Task 3 Test Results -> Macro-F1: {tag_metrics['macro_f1']:.4f}, AUC-PR: {tag_metrics['mean_auc_pr']:.4f}, Valence MAE: {emo_metrics['mae']:.4f}")

    # Generate t-SNE and Case Studies
    if len(all_z) > 0:
        z_mat = np.vstack(all_z)
        plot_tsne_embeddings(z_mat, [f"Class {np.argmax(row)}" for row in y_true])

    generate_case_studies(model, test_ds, device=device)

    return {**tag_metrics, "emotion_mae": emo_metrics["mae"]}


def train_task_4(args, cfg, device):
    """Task 4: Contrastive Dual-Encoder (MusicCaps)."""
    print("\n--- Training Task 4: Contrastive Dual-Encoder (InfoNCE) ---")
    train_ds = MusicMultimodalDataset("data/raw/dataset_metadata.json", "data/splits/splits.json", split="train")
    test_ds = MusicMultimodalDataset("data/raw/dataset_metadata.json", "data/splits/splits.json", split="test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_multimodal_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multimodal_batch)

    model = DualEncoderGNNBERT().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            graphs = batch["graphs"]
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            if hasattr(graphs, "x"):
                out = model(graphs.x.to(device), graphs.edge_index.to(device), graphs.batch.to(device), input_ids, attn_mask)
                loss = out["loss"]
            else:
                # Single pass on individual graphs then stack
                g_embeds = [
                    model.encode_graph(g.x.to(device), g.edge_index.to(device), None) for g in graphs
                ]
                g_tensor = torch.cat(g_embeds, dim=0)
                t_tensor = model.encode_text(input_ids, attn_mask)
                sim_matrix = torch.matmul(g_tensor, t_tensor.T) / model.temperature
                targets = torch.arange(len(graphs), device=device)
                loss = (torch.nn.functional.cross_entropy(sim_matrix, targets) + torch.nn.functional.cross_entropy(sim_matrix.T, targets)) / 2.0

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{args.epochs}] InfoNCE Loss: {total_loss / max(1, len(train_loader)):.4f}")

    # Evaluate Retrieval
    model.eval()
    all_g, all_t, all_caps, all_ids = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            graphs = batch["graphs"]
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            if hasattr(graphs, "x"):
                g = model.encode_graph(graphs.x.to(device), graphs.edge_index.to(device), graphs.batch.to(device))
            else:
                g = torch.cat([model.encode_graph(gx.x.to(device), gx.edge_index.to(device), None) for gx in graphs], dim=0)
            t = model.encode_text(input_ids, attn_mask)
            all_g.append(g)
            all_t.append(t)
            all_caps.extend(batch["captions"])
            all_ids.extend(batch["track_ids"])

    g_embeds = torch.cat(all_g, dim=0)
    t_embeds = torch.cat(all_t, dim=0)

    retrieval_metrics = evaluate_cross_modal_retrieval(
        g_embeds, t_embeds, all_caps, all_ids, output_qualitative_json="results/retrieval_examples/retrieval_qualitative.json"
    )
    print(f"Task 4 Retrieval Results -> Caption->Audio R@1: {retrieval_metrics['Caption_to_Audio_R1']:.4f}, R@5: {retrieval_metrics['Caption_to_Audio_R5']:.4f}")
    return retrieval_metrics


def train_baseline_cnn(args, cfg, device):
    """B2: Mel-Spectrogram CNN Baseline."""
    print("\n--- Training Baseline 2: Mel-Spectrogram CNN ---")
    train_ds = MusicMultimodalDataset("data/raw/dataset_metadata.json", "data/splits/splits.json", split="train")
    test_ds = MusicMultimodalDataset("data/raw/dataset_metadata.json", "data/splits/splits.json", split="test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_multimodal_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multimodal_batch)

    num_classes = len(train_ds.tags_vocab)
    model = MelSpectrogramCNN(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            specs = batch["spectrograms"].to(device)
            labels = batch["tag_vecs"].to(device)
            out = model(specs, labels=labels)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{args.epochs}] CNN Loss: {total_loss / max(1, len(train_loader)):.4f}")

    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            specs = batch["spectrograms"].to(device)
            out = model(specs)
            all_preds.append(out["probs"].cpu().numpy())
            all_trues.append(batch["tag_vecs"].numpy())

    y_prob = np.vstack(all_preds)
    y_true = np.vstack(all_trues)
    metrics = compute_classification_metrics(y_true, y_prob)
    print(f"CNN Baseline Results -> Macro-F1: {metrics['macro_f1']:.4f}, AUC-PR: {metrics['mean_auc_pr']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate models for GNN-BERT music context.")
    parser.add_argument("--task", type=str, default="all", choices=["1", "2", "3", "4", "baseline_cnn", "all"])
    parser.add_argument("--gnn_type", type=str, default="GraphSAGE", choices=["GraphSAGE", "GAT"])
    parser.add_argument("--fusion_type", type=str, default="cross_attention", choices=["cross_attention", "early_concat", "bert_only", "gnn_only"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("project", {}).get("device") == "cuda" else "cpu")
    print(f"Using compute device: {device}")

    # Ensure sample dataset exists
    if not os.path.exists("data/splits/splits.json"):
        print("Data splits not found. Generating sample benchmark dataset...")
        from sample_data_gen import generate_sample_dataset
        generate_sample_dataset("data", num_samples=30)

    results_dict = {}

    if args.task in ["1", "all"]:
        m1 = train_task_1(args, cfg, device)
        results_dict["Task 1: BERT-only"] = {"Macro-F1": m1["macro_f1"], "AUC-PR": m1["mean_auc_pr"]}

    if args.task in ["2", "all"]:
        m2 = train_task_2(args, cfg, device)
        results_dict["Task 2: GNN-only"] = {"Macro-F1": m2["macro_f1"], "AUC-PR": m2["mean_auc_pr"]}

    if args.task in ["baseline_cnn", "all"]:
        m_cnn = train_baseline_cnn(args, cfg, device)
        results_dict["CNN mel-spec"] = {"Macro-F1": m_cnn["macro_f1"], "AUC-PR": m_cnn["mean_auc_pr"]}

    if args.task in ["3", "all"]:
        m3 = train_task_3(args, cfg, device)
        results_dict["Task 3: GNN-BERT"] = {
            "Macro-F1": m3["macro_f1"],
            "AUC-PR": m3["mean_auc_pr"],
            "MAE (emotion)": m3.get("emotion_mae", 0.85),
        }

    if args.task in ["4", "all"]:
        m4 = train_task_4(args, cfg, device)
        results_dict["Task 4: Contrastive"] = {
            "R@5 (retrieval)": m4["Caption_to_Audio_R5"],
            "R@1 (retrieval)": m4["Caption_to_Audio_R1"],
        }

    # Save metrics table
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    print("\nSaved comprehensive results to results/metrics.json")


if __name__ == "__main__":
    main()
