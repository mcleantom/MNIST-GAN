import typer
from torchvision import datasets, transforms
from pathlib import Path
from typing import Optional, Tuple
import random
import numpy as np
import torch
import pytorch_lightning as pl

from mnist_gan.data_module import MNISTDataModule, DEFAULT_DATA_DIR, BATCH_SIZE, NUM_WORKERS
from mnist_gan.model import GAN

from pytorch_lightning import loggers as pl_loggers

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
    n_batch: int = BATCH_SIZE,
    seed: int = 7,
    lr: float = 0.001,
    max_epochs: int = 100,
    data_dir: Path = DEFAULT_DATA_DIR,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tensorboard = pl_loggers.TensorBoardLogger('./')
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
        logger=tensorboard
    )
    trainer.fit(model, data)
    model.plot_imgs()

@app.command("plot-checkpoint")
def plot_checkpoint(
        checkpoint_path: Optional[Path] = None,
):
    if checkpoint_path is None:
        lightning_logs = Path.cwd() / "lightning_logs"
        latest_version = sorted(lightning_logs.glob("version_*"))[-1]
        latest_checkpoint = sorted(latest_version.glob("checkpoints/*.ckpt"))[-1]
        checkpoint_path = latest_checkpoint
    model = GAN.load_from_checkpoint(checkpoint_path)
    model.plot_imgs()

if __name__ == "__main__":
    app()
