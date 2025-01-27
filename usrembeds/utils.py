import datetime
import torch
from datautils.dataset import MusicDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse


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


def gen_run_name(args=None):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    return run_name


def get_args():
    """
    Function to get the arguments from the command line

    Returns:
    - args (dict): arguments
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="""Get the params""",
    )

    parser.add_argument(
        "-E",
        "--emb_size",
        type=int,
        help="User embdedding size",
        default=100,
    )

    parser.add_argument(
        "-B",
        "--batch_size",
        type=int,
        help="Batch size",
        default=16,
    )

    parser.add_argument(
        "-N",
        "--neg_samples",
        type=int,
        help="Number of negative samples",
        default=20,
    )

    parser.add_argument(
        "-T",
        "--temp",
        type=float,
        help="Temperature for the InfoNCE loss",
        default=0.07,
    )

    parser.add_argument(
        "-M",
        "--multiplier",
        type=int,
        help="Dataset multiplier",
        default=10,
    )

    parser.add_argument(
        "-NL",
        "--no-log",
        action="store_true",
        help="Don't log via wandb",
        default=False,
    )

    parser.add_argument(
        "-LT",
        "--learnable-temp",
        action="store_true",
        help="Make temp a learnable parameter",
        default=False,
    )

    parser.add_argument(
        "-W",
        "--weight",
        type=float,
        help="Weight for the loss",
        default=0,
    )

    parser.add_argument(
        "-P",
        "--prj",
        type=str,
        help="Define the projection type [bn, ln, linear]",
        default="bn",
    )

    parser.add_argument(
        "--prj-size",
        type=int,
        help="Size of the latent space",
        default=768,
    )

    parser.add_argument(
        "-L",
        "--load",
        type=str,
        help="Load model and corresponding config from a checkpoint",
        default="",
    )

    parser.add_argument(
        "--encoder",
        type=str,
        help="Encoder user for music embeddings [ol3, mert]",
        default="ol3",
    )

    parser.add_argument(
        "-D",
        "--drop",
        type=float,
        help="Droupout rate",
        default=0.35,
    )

    parser.add_argument(
        "-LR",
        "--lr",
        type=float,
        help="Learning rate",
        default=0.001,
    )

    args = parser.parse_args()
    args_dict = vars(args)
    return args, args_dict
