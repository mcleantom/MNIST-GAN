import typer
from torchvision import datasets, transforms
from pathlib import Path
from typing import Optional, Tuple
import random
import numpy as np
import torch
import pytorch_lightning as pl

from mnist_gan.data_module import MNISTDataModule
from mnist_gan.model import GAN

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"

app = typer.Typer()

def get_accelerator_and_gpus() -> Tuple[str, int]:
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu"))
    if device == "cuda":
        accelerator = "gpu"
        gpus = min(1, torch.cuda.device_count())
    else:
        accelerator = "cpu"
        gpus = 0
    return accelerator, gpus


@app.command()
def main(
    n_batch: int = 256,
    seed: int = 42,
    lr: float = 0.001,
    max_epochs: int = 100,
    data_dir: Path = DEFAULT_DATA_DIR,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = MNISTDataModule(
        batch_size=n_batch,
        data_dir=data_dir
    )
    model = GAN(lr=lr)
    model.plot_imgs()
    accelerator, gpus = get_accelerator_and_gpus()
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
    )
    trainer.fit(model, data)
    model.plot_imgs()



@app.command()
def get_mnist(download_dir: Path = DEFAULT_DATA_DIR):
    # if download path, assert it is a directory
    if download_dir is not None:
        if not download_dir.exists():
            download_dir.mkdir()
        if not download_dir.is_dir():
            raise ValueError("download_dir must be a directory")
    typer.echo("Downloading MNIST dataset")
    MNISTDataModule(
        batch_size=None,
        data_dir=download_dir
    ).prepare_data()


if __name__ == "__main__":
    app()
