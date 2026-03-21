"""
Full anomaly detection pipeline.
Steps 1-3: Windowing (drop first 30, take 10 steady-state per sensor×HS)
Step 4: Gate bad sensors
Steps 5-6: Calibration (Normal Air baseline) + normalization
Step 7: Repeat across all p blocks and substances
Steps 8-9: 80 plots raw + 80 normalized per substance
Step 10: Global dim reduction (PCA/t-SNE/UMAP)
Step 11: Anomaly scoring — model trained on Normal Air, evaluated on all
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")

def run():
    import numpy as np
    from collections import defaultdict
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore")
    from data_utils import (
        BASE_DIR, SUBSTANCES, N_SENSORS, N_STEPS, N_WARMUP, M_STEADY,
        TARGET_SAMPLES_PER_SUBSTANCE, WINDOW_STRIDE,
        load_folder_blocks, compute_baseline, compute_stats, apply_windowing,
        apply_windowing_multi, normalize_block, extract_features, get_valid_sensors,
    )
    SUBSTANCE_PREFIX = {s: s for s in SUBSTANCES}

    print("Steps 1-3: Load + windowing (n=30 warmup, m=10 steady)...")
    all_raw_blocks = {}
    for sub in SUBSTANCES:
        folder = BASE_DIR / sub
        if not folder.exists():
            continue
        blocks, labels = load_folder_blocks(folder, "Normal_Air_*")
        if not blocks:
            blocks, labels = load_folder_blocks(folder, "Normal*")
        windowed = [apply_windowing(b, n=N_WARMUP, m=M_STEADY) for b in blocks]
        all_raw_blocks[sub] = (windowed, labels)

    print("Step 5: Calibration (Normal Air baseline)...")
    calibration = {}
    for sub in SUBSTANCES:
        if sub not in all_raw_blocks:
            continue
        windowed, _ = all_raw_blocks[sub]
        calibration[sub] = compute_baseline(windowed, apply_windowing_flag=False)

    print("Steps 4, 6-7: Gate sensors, normalize, extract features (sliding windows for ~300/substance)...")
    def load_substance_data(sub):
        folder = BASE_DIR / sub
        prefix = SUBSTANCE_PREFIX.get(sub, sub)
        norm_blks, norm_lbls = all_raw_blocks.get(sub, ([], []))
        sub_blocks, sub_labels = load_folder_blocks(folder, f"{prefix}_*")
        # Multi-window: each raw block → multiple windowed sub-blocks (for ~300 samples)
        sub_windowed, sub_windowed_labels = [], []
        for b, lbl in zip(sub_blocks, sub_labels):
            for i, w in enumerate(apply_windowing_multi(b, stride=WINDOW_STRIDE)):
                sub_windowed.append(w)
                sub_windowed_labels.append(f"{lbl}_w{i}" if i > 0 else lbl)
        return norm_blks, norm_lbls, sub_windowed, sub_windowed_labels

    def process_blocks(blocks, labels, baseline, tag):
        raw, norm, valid_list = defaultdict(list), defaultdict(list), []
        for blk, lbl in zip(blocks, labels):
            vs = get_valid_sensors(blk)
            if len(vs) < 4:
                continue
            valid_list.append((blk, lbl, vs))
            nb = normalize_block(blk, baseline)
            for (s, st), vals in blk.items():
                if s in vs and vals:
                    raw[(s, st)].extend(vals)
            for (s, st), arr in nb.items():
                if s in vs and len(arr) > 0:
                    norm[(s, st)].extend(arr.tolist())
        return raw, norm, valid_list

    all_features, all_labels, block_labels = [], [], []
    rng = np.random.default_rng(42)
    for sub in SUBSTANCES:
        if sub not in calibration:
            continue
        norm_blks, norm_lbls, sub_blks, sub_lbls = load_substance_data(sub)
        baseline = calibration[sub]
        for blocks, labels, tag in [(norm_blks, norm_lbls, "Normal_Air"), (sub_blks, sub_lbls, sub)]:
            if not blocks:
                continue
            if len(labels) != len(blocks):
                labels = [f"b{i}" for i in range(len(blocks))]
            _, _, valid_blocks = process_blocks(blocks, labels, baseline, tag)
            indices = list(range(len(valid_blocks)))
            # Cap substance samples at TARGET_SAMPLES_PER_SUBSTANCE (300)
            if tag != "Normal_Air" and len(indices) > TARGET_SAMPLES_PER_SUBSTANCE:
                indices = rng.choice(len(valid_blocks), TARGET_SAMPLES_PER_SUBSTANCE, replace=False)
            for i in indices:
                blk, lbl, vs = valid_blocks[i]
                all_features.append(extract_features(blk, baseline, vs))
                all_labels.append(tag)
                block_labels.append(f"{tag}_{lbl}")

    X = np.vstack(all_features)
    y = np.array(all_labels)
    print(f"  X: {X.shape}, blocks: {len(all_labels)}")

    print("Steps 8-9: 80 plots raw + normalized per substance...")
    OUTPUT_DIR = BASE_DIR / "pipeline_output"
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "stats_plots").mkdir(exist_ok=True)

    def plot_80_grid(data, title, filepath, stat_key="median"):
        fig, axes = plt.subplots(8, 10, figsize=(18, 12))
        for s in range(N_SENSORS):
            for st in range(N_STEPS):
                ax = axes[s, st]
                vals = data.get((s, st), [])
                if vals:
                    stats = compute_stats(np.array(vals))
                    v = stats.get(stat_key, 0)
                    v = 0 if (np.isnan(v) or np.isinf(v)) else float(v)
                    ax.bar([stat_key], [v], color="steelblue")
                ax.set_title(f"S{s} HS{st}", fontsize=7)
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches="tight")
        plt.close()

    for sub in SUBSTANCES:
        if sub not in calibration:
            continue
        norm_blks, _, sub_blks, _ = load_substance_data(sub)
        combined = norm_blks + sub_blks
        raw_r, norm_r, _ = process_blocks(
            combined, [""] * len(combined), calibration[sub], sub
        )
        plot_80_grid(raw_r, f"{sub} Raw (m=10)", OUTPUT_DIR / f"stats_plots/{sub}_raw.png")
        plot_80_grid(norm_r, f"{sub} Normalized", OUTPUT_DIR / f"stats_plots/{sub}_norm.png")
    print("  Saved")

    print("Step 10: PCA / t-SNE / UMAP (substances only, Normal Air excluded)...")
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = StandardScaler().fit_transform(X_clean)

    # Exclude Normal Air from dim reduction — substances only
    sub_mask = y != "Normal_Air"
    X_sub = X_scaled[sub_mask]
    y_sub = y[sub_mask]

    X_pca = PCA(n_components=2, random_state=42).fit_transform(X_sub)
    perplexity = min(30, max(5, len(X_sub) - 1))
    X_tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(X_sub)
    try:
        import umap
        X_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(X_sub)
    except ImportError:
        X_umap = X_pca

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, emb, name in zip(axes, [X_pca, X_tsne, X_umap], ["PCA", "t-SNE", "UMAP"]):
        for lab in np.unique(y_sub):
            mask = y_sub == lab
            ax.scatter(emb[mask, 0], emb[mask, 1], label=lab, alpha=0.6, s=40)
        ax.set_title(name)
        ax.legend(fontsize=8)
    plt.suptitle("Step 10: Dimensionality reduction (substances only)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dim_reduction.png", dpi=120, bbox_inches="tight")
    plt.close()

    print("Step 11: Anomaly scoring (train on Normal Air, evaluate on all)...")
    from sklearn.ensemble import IsolationForest

    normal_mask = y == "Normal_Air"
    X_normal = X_scaled[normal_mask]
    X_all = X_scaled

    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_normal)
    scores = model.decision_function(X_all)  # higher = more normal
    preds = model.predict(X_all)  # 1=normal, -1=anomaly

    # Per-substance summary
    results = []
    for lab in np.unique(y):
        mask = y == lab
        n = mask.sum()
        n_anomaly = (preds[mask] == -1).sum()
        mean_score = scores[mask].mean()
        results.append((lab, n, n_anomaly, 100 * n_anomaly / n, mean_score))

    out_path = OUTPUT_DIR / "anomaly_scores.csv"
    with open(out_path, "w") as f:
        f.write("substance,n_blocks,n_anomalies,pct_anomalous,mean_decision_score\n")
        for lab, n, na, pct, ms in results:
            f.write(f"{lab},{n},{na},{pct:.1f},{ms:.4f}\n")
    print("  Results:")
    for lab, n, na, pct, ms in results:
        print(f"    {lab}: {n} blocks, {na} anomalies ({pct:.1f}%), mean_score={ms:.3f}")
    print(f"  Saved {out_path}")

    # Anomaly score distribution plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for lab in np.unique(y):
        mask = y == lab
        ax.hist(scores[mask], bins=20, alpha=0.5, label=lab, density=True)
    ax.axvline(0, color="black", linestyle="--", label="decision boundary")
    ax.set_xlabel("Anomaly score (higher = more normal)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("Step 11: Anomaly score distribution by substance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "anomaly_scores.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved anomaly_scores.png")
    print("\nDone. Output in pipeline_output/")

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
