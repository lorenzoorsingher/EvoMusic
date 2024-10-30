import torch
from datautils.dataset import MusicDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_music_batch(emb, device):

    emb_flat = emb.view(-1, emb.shape[2])  # flatten the first two dimensions
    tsne = TSNE(n_components=2)

    emb_2d = tsne.fit_transform(emb_flat.cpu().detach().numpy())
    emb_2d = torch.tensor(emb_2d, device=device).view(
        emb.shape[0], emb.shape[1], 2
    )  # reshape back to [16, 6, 2]

    emb_2d_np = emb_2d.cpu().detach().numpy()

    plt.figure(figsize=(10, 8))
    for i in range(emb_2d_np.shape[0]):
        cluster_points = emb_2d_np[i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i+1}")

    plt.title("t-SNE Embedding")
    plt.show()
