import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from umap import UMAP  # Updated import

# --- Import your EvoMusic components ---
from EvoMusic.configuration import load_yaml_config
from EvoMusic.music_generation.generators import MusicGenPipeline, EasyRiffPipeline
from EvoMusic.user_embs.model import AlignerV2
from EvoMusic.evolution.fitness import MusicScorer


def visualize_scatter(embeds, labels, title="Scatter Plot", label_set=None, projection='3d', color_by='genre'):
    """
    Utility to plot a scatter plot (2D or 3D) of the given embeddings.

    Parameters:
    - embeds: np.array of shape (N, 2) or (N, 3)
    - labels: list of length N, labeling each point
    - title: Title of the plot
    - label_set: optional ordered list of unique labels
    - projection: '3d' or '2d'
    - color_by: 'genre' or 'density'
    """
    if label_set is None and color_by == 'genre':
        label_set = sorted(list(set(labels)))

    fig = plt.figure(figsize=(10, 8))
    if projection == '3d':
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    ax.set_title(title, fontsize=15)

    if color_by == 'genre':
        # Assign each unique label a consistent color using a qualitative colormap
        label_to_color = {}
        cmap = plt.get_cmap('tab10', len(label_set))  # Use 'tab10' for better distinction
        for i, lab in enumerate(label_set):
            label_to_color[lab] = cmap(i)

        colors = [label_to_color[lab] for lab in labels]
        if projection == "3d":
            scatter = ax.scatter(
                embeds[:, 0], embeds[:, 1], embeds[:, 2],
                c=colors,
                s=50, alpha=0.7, edgecolors='w', linewidth=0.5
            )
        else:
            scatter = ax.scatter(
                embeds[:, 0], embeds[:, 1],
                c=colors,
                s=50, alpha=0.7, edgecolors='w', linewidth=0.5
            )

        # Create legend with distinct markers for centroids
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', 
                       label=lab, markerfacecolor=label_to_color[lab],
                       markersize=10, markeredgecolor='k')
            for lab in label_set
        ]
        plt.legend(
            handles=handles,
            title="Genres",
            loc="best",
            fontsize=10,
            bbox_to_anchor=(1.05, 1),
            borderaxespad=0.
        )
    elif color_by == 'density':
        # Compute density
        if projection == '3d':
            density = compute_density(embeds)
        else:
            density = compute_density(embeds[:, :2])

        # Normalize density for coloring
        density_normalized = (density - density.min()) / (density.max() - density.min())

        # Choose a colormap
        cmap = plt.cm.viridis
        colors = cmap(density_normalized)

        if projection == '3d':
            scatter = ax.scatter(
                embeds[:, 0], embeds[:, 1], embeds[:, 2],
                c=density_normalized,
                cmap=cmap,
                s=50, alpha=0.7, edgecolors='w'
            )
            cb = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
            cb.set_label('Density')
        else:
            scatter = ax.scatter(
                embeds[:, 0], embeds[:, 1],
                c=density_normalized,
                cmap=cmap,
                s=50, alpha=0.7, edgecolors='w'
            )
            cb = plt.colorbar(scatter, ax=ax)
            cb.set_label('Density')

    if projection == '3d':
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.set_zlabel("Z", fontsize=12)
    else:
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)

    plt.tight_layout()
    plt.show()


def compute_density(embeds, bandwidth=1.0):
    """
    Compute the density of each point using Kernel Density Estimation.

    Parameters:
    - embeds: np.array of shape (N, D)
    - bandwidth: float, bandwidth for KDE

    Returns:
    - density: np.array of shape (N,), normalized between 0 and 1
    """
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(embeds)
    log_density = kde.score_samples(embeds)
    density = np.exp(log_density)
    # Normalize density
    density = (density - density.min()) / (density.max() - density.min())
    return density


def visualize_class_distribution_gmm(embeds_2d, labels, class_list=None,
                                     n_components=1, grid_size=200,
                                     title_prefix=""):
    """
    Show a side-by-side 2D visualization:
      (A) Scatter plot of points with class centroids
      (B) Fractional density map from multi-class GMM

    Args:
        embeds_2d: np.ndarray of shape (N, 2), the 2D embedding for each sample.
        labels: list of length N, each an integer or string label for the sample's class.
        class_list: optional list specifying the unique classes. If None, derived from labels.
        n_components: how many Gaussian components to use per class GMM.
        grid_size: resolution for the fractional density grid.
        title_prefix: a string to prepend to the figure titles.
    """
    if class_list is None:
        class_list = sorted(list(set(labels)))

    # Convert class labels to numeric indices if needed
    # If your labels are already strings, create a mapping
    label_to_idx = {}
    for i, c in enumerate(class_list):
        label_to_idx[c] = i
    numeric_labels = np.array([label_to_idx[l] for l in labels], dtype=int)

    # Fit one GMM per class
    gmms = []
    for cidx, c in enumerate(class_list):
        points_c = embeds_2d[numeric_labels == cidx]
        if len(points_c) == 0:
            print(f"Warning: No points found for class '{c}'. Skipping GMM fitting.")
            gmms.append(None)
            continue
        gmm_c = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm_c.fit(points_c)
        gmms.append(gmm_c)

    # Compute class centroids
    centroids = []
    for cidx, c in enumerate(class_list):
        points_c = embeds_2d[numeric_labels == cidx]
        if len(points_c) == 0:
            centroid_c = np.array([np.nan, np.nan])
        else:
            centroid_c = points_c.mean(axis=0)
        centroids.append(centroid_c)

    centroids = np.array(centroids)  # shape (#classes, 2)

    # Build a color palette for classes using a qualitative colormap
    cmap = plt.get_cmap('tab10', len(class_list))  # 'tab10' has 10 distinct colors
    class_colors = [cmap(i) for i in range(len(class_list))]

    # Create a grid covering the embedding range
    x_min, x_max = embeds_2d[:, 0].min(), embeds_2d[:, 0].max()
    y_min, y_max = embeds_2d[:, 1].min(), embeds_2d[:, 1].max()

    margin_x = 0.05 * (x_max - x_min)
    margin_y = 0.05 * (y_max - y_min)
    x_min -= margin_x
    x_max += margin_x
    y_min -= margin_y
    y_max += margin_y

    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)
    XX, YY = np.meshgrid(xs, ys)
    grid_points = np.column_stack((XX.ravel(), YY.ravel()))  # shape (grid_size^2, 2)

    # Evaluate each class GMM at each grid point
    pdfs = []
    for cidx, gmm_c in enumerate(gmms):
        if gmm_c is None:
            pdf_c = np.zeros(grid_points.shape[0])
        else:
            # pdf_c: shape (grid_size^2,)
            pdf_c = np.exp(gmm_c.score_samples(grid_points))
        pdfs.append(pdf_c)

    pdfs = np.array(pdfs)  # shape (#classes, grid_size^2)
    sum_pdfs = pdfs.sum(axis=0) + 1e-12  # avoid division by zero
    frac_pdfs = pdfs / sum_pdfs  # shape (#classes, grid_size^2)

    # For each grid point, blend class colors by fractional membership
    # color_at_grid = sum_i( frac_pdfs[i] * class_colors[i] )
    # class_colors[i] is RGBA, frac_pdfs[i,j] is fraction
    blended_colors = np.zeros((grid_size * grid_size, 4))
    for i in range(len(class_list)):
        color_i = np.array(class_colors[i])  # RGBA
        # Add fraction * color_i
        for c in range(4):
            blended_colors[:, c] += frac_pdfs[i, :] * color_i[c]

    # Reshape to 2D grid for imshow
    blended_colors = blended_colors.reshape((grid_size, grid_size, 4))

    # Build side-by-side figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f"{title_prefix} Class Distribution via GMM", fontsize=20)

    # ---------------------------------------------
    # Left: scatter of points + centroids
    # ---------------------------------------------
    ax0 = axes[0]
    ax0.set_title("Samples + Centroids", fontsize=16)
    # Plot each class
    for cidx, c in enumerate(class_list):
        points_c = embeds_2d[numeric_labels == cidx]
        if len(points_c) == 0:
            continue
        ax0.scatter(points_c[:, 0], points_c[:, 1],
                    color=class_colors[cidx],
                    label=str(c), alpha=0.6, edgecolors='k', s=40)

    # Plot centroids with distinct markers
    valid_centroids = ~np.isnan(centroids).any(axis=1)
    ax0.scatter(centroids[valid_centroids, 0], centroids[valid_centroids, 1],
                s=200, c='black', marker='X', label='Centroids', edgecolors='w', linewidth=2)

    ax0.set_xlabel("X", fontsize=14)
    ax0.set_ylabel("Y", fontsize=14)
    ax0.legend(loc="best", fontsize=10, markerscale=1.5, frameon=True)

    # ---------------------------------------------
    # Right: blended fractional color map
    # ---------------------------------------------
    ax1 = axes[1]
    ax1.set_title("Fractional Density Map", fontsize=16)
    ax1.imshow(
        blended_colors,
        origin='lower',  # so that Y=0 is at the bottom
        extent=(x_min, x_max, y_min, y_max),
        aspect='auto'
    )
    # Overlay original points as semi-transparent white
    ax1.scatter(embeds_2d[:, 0], embeds_2d[:, 1],
                color='white', edgecolors='k', s=20, alpha=0.3)
    ax1.set_xlabel("X", fontsize=14)
    ax1.set_ylabel("Y", fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main(
    json_file="prompts.json",
    yaml_config="example_conf/test_music_generation_config.yaml",
    output_dir="generated_audio",
    device="cuda",
    max_genres=None,
    max_prompts_per_genre=None
):
    """
    1) Loads config and JSON prompts.
    2) Generates audio for each prompt.
    3) Embeds audio using MERT (via MusicScorer).
    4) Optionally passes embeddings through AlignerV2.
    5) Visualizes embeddings in 3D and 2D via t-SNE & UMAP with color gradients based on class density.

    Parameters:
    - json_file: Path to the JSON file containing prompts.
    - yaml_config: Path to the YAML configuration file.
    - output_dir: Directory where generated audio will be saved.
    - device: Computation device ('cuda' or 'cpu').
    - max_genres: Maximum number of genres to process (None means no limit).
    - max_prompts_per_genre: Maximum number of prompts to process per genre (None means no limit).
    """
    # -----------------------
    # Load evolution config
    # -----------------------
    config = load_yaml_config(yaml_config)
    config.device = device
    os.makedirs(output_dir, exist_ok=True)

    # Initialize your MusicScorer (which wraps MERT)
    music_scorer = MusicScorer(config.evolution.fitness)

    # -----------------------
    # Initialize generator
    # -----------------------
    if config.music_model == "musicgen":
        generator_config = config.music_generator
        gen = MusicGenPipeline(generator_config)
    else:
        generator_config = config.riffusion_pipeline
        gen = EasyRiffPipeline(generator_config)

    # -----------------------
    # Optionally load AlignerV2
    # (Only if you want "post-aligner" embeddings)
    # -----------------------
    aligner_path = "usrembeds/checkpoints/AlignerV2_best_model.pt"
    if os.path.exists(aligner_path):
        try:
            state_dict, aligner_conf, _ = AlignerV2.load_model(aligner_path, device=device)
            model_conf = {
                "n_users": aligner_conf["nusers"],
                "emb_size": aligner_conf["emb_size"],
                "prj_size": aligner_conf["prj_size"],
                "hidden_size": 2048,
                "prj_type": aligner_conf["prj"],
                "aggragation": aligner_conf["aggr"],
                "noise_level": 0.0,
                "lt": aligner_conf["learnable_temp"],
                "temp": 0.07,
                "drop": 0.35,
            }
            aligner = AlignerV2(**model_conf).to(device)
            aligner.load_state_dict(state_dict)
            aligner.eval()
            print("AlignerV2 model loaded successfully.")
        except Exception as e:
            print(f"Error loading AlignerV2 model: {e}")
            aligner = None
    else:
        print(f"AlignerV2 model not found at {aligner_path}. Proceeding without it.")
        aligner = None

    # -----------------------
    # Read JSON prompts
    # Example structure:
    # {
    #   "music_genre": [
    #       {"id": 0, "prompt": "some text prompt"},
    #       ...
    #   ],
    #   "pop": [
    #       {"id": 0, "prompt": "another prompt"}, ...
    #   ],
    #   ...
    # }
    # -----------------------
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            prompts_dict = json.load(f)
        print(f"Loaded prompts from {json_file}.")
    except Exception as e:
        print(f"Failed to load JSON file {json_file}: {e}")
        return

    # Flatten out all (genre, prompt) pairs with limits
    all_prompts = []
    genres = list(prompts_dict.keys())
    if max_genres is not None:
        genres = genres[:max_genres]
        print(f"Limiting to first {max_genres} genres.")

    for genre in genres:
        items = prompts_dict[genre]
        if max_prompts_per_genre is not None:
            items = items[:max_prompts_per_genre]
            print(f"Limiting genre '{genre}' to first {max_prompts_per_genre} prompts.")
        for entry in items:
            p = entry["prompt"]
            all_prompts.append((genre, p))

    print(f"Total prompts to process: {len(all_prompts)}")

    # For collecting results
    all_audio_paths = []
    all_genres = []

    # -----------------------
    # Generate audio & collect paths
    # -----------------------
    print("Generating audio from prompts (this may be slow)...")
    for (genre, prompt) in tqdm(all_prompts, desc="Generating Music"):
        try:
            audio_path = gen.generate_music(prompt, duration=2)  # Adjust duration as needed
            all_audio_paths.append(audio_path)
            all_genres.append(genre)
        except Exception as e:
            print(f"Failed to generate music for prompt '{prompt}' in genre '{genre}': {e}")

    print("Done generating. Now embedding them with MERT...")

    # -----------------------
    # Embed with MERT
    # This yields shape [num_files, 13, *hidden_dim*]
    # -----------------------
    try:
        with torch.no_grad():
            mert_embs_4d = music_scorer.embed_audios(all_audio_paths).to(device)
        print(f"Embedded {mert_embs_4d.shape[0]} audio files with MERT.")
    except Exception as e:
        print(f"Failed to embed audios with MERT: {e}")
        return

    # We'll build a "pre-aligner" 2D array for dimensionality reduction
    # For example, just flatten across the 13 frames -> shape (N, 13 * emb_size)
    # (MERT's base model dimension is ~768; if so, shape -> (N, 13*768 = 9984). That might be large, so be mindful.)
    try:
        N, L, D = mert_embs_4d.shape
        mert_embs_flat = mert_embs_4d.reshape(N, L * D).cpu().numpy()  # (N, 13*D)
        print(f"Flattened MERT embeddings to shape {mert_embs_flat.shape}.")
    except Exception as e:
        print(f"Failed to flatten MERT embeddings: {e}")
        return

    # -----------------------
    # (Optional) Post-Aligner Embeddings
    # Aligners typically want shape (batch_size, N_pos+N_neg, 13, emb_size).
    # For a simple case, treat each sample as a "batch" with 1 example:
    # We'll also fake a user index (e.g. 0) if you're only visualizing embeddings.
    # -----------------------
    if aligner is not None:
        try:
            user_idx = torch.tensor([0], device=device, dtype=torch.long)  # single user
            # shape must be (batch=1, N=number_of_samples, 13, D)
            mert_embs_for_aligner = mert_embs_4d.unsqueeze(0)  # (1, N, 13, D)

            with torch.no_grad():
                usr_x, music_x, temp = aligner(user_idx, mert_embs_for_aligner)
                # music_x: (1, N, prj_size)
                # Squeeze the batch dimension
                aligner_embs = music_x.squeeze(0)  # (N, prj_size)
                post_aligner_embs = aligner_embs.cpu().numpy()
            print("Obtained Post-Aligner embeddings.")
        except Exception as e:
            print(f"Failed to obtain Post-Aligner embeddings: {e}")
            post_aligner_embs = None
    else:
        post_aligner_embs = None

    # -----------------------
    # Now we do T-SNE and UMAP on both sets of embeddings
    # (1) MERT's raw 13-layer flatten
    # (2) Post-Aligner
    # We will project each to 3D and 2D and do scatter plots with color gradients based on density.
    # -----------------------
    print("Running dimensionality reduction... (t-SNE & UMAP)")

    # ---- t-SNE on MERT (pre-aligner) ----
    try:
        print("Applying t-SNE on MERT embeddings (3D)...")
        tsne_3d_pre = TSNE(n_components=3, perplexity=15, init='pca', random_state=42)
        mert_tsne_3d = tsne_3d_pre.fit_transform(mert_embs_flat)
        print("t-SNE 3D on MERT embeddings completed.")
    except Exception as e:
        print(f"t-SNE 3D on MERT embeddings failed: {e}")
        mert_tsne_3d = None

    try:
        print("Applying t-SNE on MERT embeddings (2D)...")
        tsne_2d_pre = TSNE(n_components=2, perplexity=15, init='pca', random_state=42)
        mert_tsne_2d = tsne_2d_pre.fit_transform(mert_embs_flat)
        print("t-SNE 2D on MERT embeddings completed.")
    except Exception as e:
        print(f"t-SNE 2D on MERT embeddings failed: {e}")
        mert_tsne_2d = None

    # ---- UMAP on MERT (pre-aligner) ----
    try:
        print("Applying UMAP on MERT embeddings (3D)...")
        umap_3d_pre = UMAP(n_neighbors=15, n_components=3, random_state=42)
        mert_umap_3d = umap_3d_pre.fit_transform(mert_embs_flat)
        print("UMAP 3D on MERT embeddings completed.")
    except Exception as e:
        print(f"UMAP 3D on MERT embeddings failed: {e}")
        mert_umap_3d = None

    try:
        print("Applying UMAP on MERT embeddings (2D)...")
        umap_2d_pre = UMAP(n_neighbors=15, n_components=2, random_state=42)
        mert_umap_2d = umap_2d_pre.fit_transform(mert_embs_flat)
        print("UMAP 2D on MERT embeddings completed.")
    except Exception as e:
        print(f"UMAP 2D on MERT embeddings failed: {e}")
        mert_umap_2d = None

    # -----------------------
    # Visualize MERT Embeddings
    # -----------------------
    # Genre-based 3D t-SNE
    if mert_tsne_3d is not None:
        visualize_scatter(
            mert_tsne_3d, all_genres, title="MERT Pre-Aligner (t-SNE 3D)", projection='3d', color_by='genre'
        )
    else:
        print("Skipping t-SNE 3D visualization for MERT embeddings due to previous errors.")

    # Genre-based 2D t-SNE
    if mert_tsne_2d is not None:
        visualize_scatter(
            mert_tsne_2d, all_genres, title="MERT Pre-Aligner (t-SNE 2D)", projection='2d', color_by='genre'
        )
    else:
        print("Skipping t-SNE 2D visualization for MERT embeddings due to previous errors.")

    # Genre-based 3D UMAP
    if mert_umap_3d is not None:
        visualize_scatter(
            mert_umap_3d, all_genres, title="MERT Pre-Aligner (UMAP 3D)", projection='3d', color_by='genre'
        )
    else:
        print("Skipping UMAP 3D visualization for MERT embeddings due to previous errors.")

    # Genre-based 2D UMAP
    if mert_umap_2d is not None:
        visualize_scatter(
            mert_umap_2d, all_genres, title="MERT Pre-Aligner (UMAP 2D)", projection='2d', color_by='genre'
        )
    else:
        print("Skipping UMAP 2D visualization for MERT embeddings due to previous errors.")

    # -----------------------
    # If we have Post-Aligner embeddings, do the same
    # -----------------------
    if post_aligner_embs is not None:
        try:
            print("Applying t-SNE on Post-Aligner embeddings (3D)...")
            tsne_3d_post = TSNE(n_components=3, perplexity=15, init='pca', random_state=42)
            aligner_tsne_3d = tsne_3d_post.fit_transform(post_aligner_embs)
            print("t-SNE 3D on Post-Aligner embeddings completed.")
        except Exception as e:
            print(f"t-SNE 3D on Post-Aligner embeddings failed: {e}")
            aligner_tsne_3d = None

        try:
            print("Applying t-SNE on Post-Aligner embeddings (2D)...")
            tsne_2d_post = TSNE(n_components=2, perplexity=15, init='pca', random_state=42)
            aligner_tsne_2d = tsne_2d_post.fit_transform(post_aligner_embs)
            print("t-SNE 2D on Post-Aligner embeddings completed.")
        except Exception as e:
            print(f"t-SNE 2D on Post-Aligner embeddings failed: {e}")
            aligner_tsne_2d = None

        try:
            print("Applying UMAP on Post-Aligner embeddings (3D)...")
            umap_3d_post = UMAP(n_neighbors=15, n_components=3, random_state=42)
            aligner_umap_3d = umap_3d_post.fit_transform(post_aligner_embs)
            print("UMAP 3D on Post-Aligner embeddings completed.")
        except Exception as e:
            print(f"UMAP 3D on Post-Aligner embeddings failed: {e}")
            aligner_umap_3d = None

        try:
            print("Applying UMAP on Post-Aligner embeddings (2D)...")
            umap_2d_post = UMAP(n_neighbors=15, n_components=2, random_state=42)
            aligner_umap_2d = umap_2d_post.fit_transform(post_aligner_embs)
            print("UMAP 2D on Post-Aligner embeddings completed.")
        except Exception as e:
            print(f"UMAP 2D on Post-Aligner embeddings failed: {e}")
            aligner_umap_2d = None

    # -----------------------
    # Visualize Post-Aligner Embeddings
    # -----------------------
    # Genre-based 3D t-SNE
    if 'aligner_tsne_3d' in locals() and aligner_tsne_3d is not None:
        visualize_scatter(
            aligner_tsne_3d, all_genres, title="Post-Aligner (t-SNE 3D)", projection='3d', color_by='genre'
        )
    else:
        print("Skipping t-SNE 3D visualization for Post-Aligner embeddings due to previous errors.")
    # Genre-based 2D t-SNE
    if 'aligner_tsne_2d' in locals() and aligner_tsne_2d is not None:
        visualize_scatter(
            aligner_tsne_2d, all_genres, title="Post-Aligner (t-SNE 2D)", projection='2d', color_by='genre'
        )
    else:
        print("Skipping t-SNE 2D visualization for Post-Aligner embeddings due to previous errors.")
    # Genre-based 3D UMAP
    if 'aligner_umap_3d' in locals() and aligner_umap_3d is not None:
        visualize_scatter(
            aligner_umap_3d, all_genres, title="Post-Aligner (UMAP 3D)", projection='3d', color_by='genre'
        )
    else:
        print("Skipping UMAP 3D visualization for Post-Aligner embeddings due to previous errors.")
    # Genre-based 2D UMAP
    if 'aligner_umap_2d' in locals() and aligner_umap_2d is not None:
        visualize_scatter(
            aligner_umap_2d, all_genres, title="Post-Aligner (UMAP 2D)", projection='2d', color_by='genre'
        )
    else:
        print("Skipping UMAP 2D visualization for Post-Aligner embeddings due to previous errors.")
        
    # -----------------------
    # Now, visualize class distribution using GMM
    # -----------------------
    print("Visualizing class distribution using GMM...")
    # Choose a 2D embedding to use (e.g., mert_tsne_2d)
    if 'mert_tsne_2d' in locals() and mert_tsne_2d is not None:
        # **Fixed KeyError Here by Passing class_list as None or Unique Genres**
        visualize_class_distribution_gmm(
            mert_tsne_2d,
            labels=all_genres,
            class_list=None,   # Option 1: Let the function derive class_list automatically
            # Alternatively, use the line below instead of class_list=None
            # class_list=sorted(list(set(all_genres))),
            n_components=1,   # or more, if you want multiple Gaussians per class
            grid_size=200,
            title_prefix="MERT t-SNE 2D"
        )
    else:
        print("Cannot visualize GMM class distribution: 2D t-SNE embedding not available.")
    # Similarly, you can visualize for other 2D embeddings if desired
    if 'aligner_tsne_2d' in locals() and aligner_tsne_2d is not None:
        visualize_class_distribution_gmm(
            aligner_tsne_2d,
            labels=all_genres,
            class_list=None,   # Option 1: Let the function derive class_list automatically
            # class_list=sorted(list(set(all_genres))),
            n_components=1,
            grid_size=200,
            title_prefix="Post-Aligner t-SNE 2D"
        )
    else:
        print("Skipping GMM visualization for Post-Aligner embeddings due to missing 2D embedding.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize MERT and Post-Aligner embeddings in 3D and 2D using t-SNE and UMAP with color gradients based on class density."
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default="visualizations/music_generation_prompts_by_genre.json",
        help="Path to the JSON file containing music prompts categorized by genre."
    )
    parser.add_argument(
        "--yaml_config",
        type=str,
        default="example_conf/visualization.yaml",
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_audio/vis",
        help="Directory where generated audio will be saved."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Computation device to use."
    )
    parser.add_argument(
        "--max_genres",
        type=int,
        default=None,
        help="Maximum number of genres to process. If not set, all genres are processed."
    )
    parser.add_argument(
        "--max_prompts_per_genre",
        type=int,
        default=None,
        help="Maximum number of prompts to process per genre. If not set, all prompts are processed."
    )
    args = parser.parse_args()
    main(
        json_file=args.json_file,
        yaml_config=args.yaml_config,
        output_dir=args.output_dir,
        device=args.device,
        max_genres=args.max_genres,
        max_prompts_per_genre=args.max_prompts_per_genre
    )
