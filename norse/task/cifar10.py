import os
import datetime
import uuid

from absl import app
from absl import flags
from absl import logging

import torch
import torch.utils.data

from collections import namedtuple
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import pytorch_lightning as pl

import norse


class LIFConvNet(pl.LightningModule):
    def __init__(
        self, seq_length, num_channels, lr, optimizer, p, noise_scale=1e-6, lr_step=True
    ):
        super().__init__()
        self.lr = lr
        self.lr_step = lr_step
        self.optimizer = optimizer
        self.seq_length = seq_length
        self.noise_distribution = torch.distributions.uniform.Uniform(1e-8, 1e-5)

        self.rsnn = norse.torch.SequentialState(
            # Convolutional layers
            torch.nn.Conv2d(num_channels, 64, 3),  # Block 1
            norse.torch.LIFCell(p),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(2, 2, ceil_mode=True),
            torch.nn.Conv2d(64, 128, 3),  # Block 2
            norse.torch.LIFCell(p),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(2, 2, ceil_mode=True),
            torch.nn.Conv2d(128, 256, 5),  # Block 3
            norse.torch.LIFCell(p),
            torch.nn.BatchNorm2d(256),
            torch.nn.MaxPool2d(2, 2, ceil_mode=True),
            torch.nn.Flatten(1),
            # Classification
            torch.nn.Linear(1024, 128),
            norse.torch.LIFCell(p),
            torch.nn.Linear(128, 10),
            norse.torch.LICell(),
        )

    def forward(self, x):
        # X was shape (batch, time, ...) and will be (time, batch, ...)
        x = x.permute(1, 0, 2, 3, 4)
        voltages = torch.empty(*x.shape[:2], 10, device=x.device, dtype=x.dtype)
        s = None
        for ts in range(x.shape[0]):
            out, s = self.rsnn(x[ts], s)
            voltages[ts, :, :] = out

        return voltages

    # Forward pass of a single batch
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        pred = pred.max(dim=0)[0]
        loss = torch.nn.functional.cross_entropy(pred, y)
        classes = out.max(0)[0].argmax(1)
        acc = torch.eq(classes, y).sum().item() / len(y)

        self.log("Loss", loss)
        self.log("LR", self.scheduler.get_last_lr()[0])
        self.log("Acc.", acc, prog_bar=True)
        return loss

    # The testing step is the same as the training, but with test data
    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = torch.nn.functional.cross_entropy(out.max(0)[0], y)
        classes = out.max(dim=0)[0].argmax(1)
        acc = torch.eq(classes, y).sum().item() / len(y)

        self.log("Loss", loss)
        self.log("Acc.", acc)
        self.log("LR", self.scheduler.get_last_lr()[0])

    def training_epoch_end(self, outputs):
        if self.lr_step:
            self.scheduler.step()

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=1e-5
            )

        if step % FLAGS.log_interval == 0 and writer:
            _, argmax = torch.max(output, 1)
            accuracy = (target == argmax.squeeze()).float().mean()

            writer.add_scalar("Loss/train", loss.item(), step)
            writer.add_scalar("Accuracy/train", accuracy.item(), step)

            for tag, value in model.named_parameters():
                tag = tag.replace(".", "/")
                writer.add_histogram(tag, value.data.cpu().numpy(), step)
                writer.add_histogram(tag + "/grad", value.grad.data.cpu().numpy(), step)

        if FLAGS.do_plot and batch_idx % FLAGS.plot_interval == 0:
            ts = np.arange(0, FLAGS.seq_length)
            _, axs = plt.subplots(4, 4, figsize=(15, 10), sharex=True, sharey=True)
            axs = axs.reshape(-1)  # flatten
            for nrn in range(10):
                one_trace = model.voltages.detach().cpu().numpy()[:, 0, nrn]
                plt.sca(axs[nrn])
                plt.plot(ts, one_trace)
            plt.xlabel("Time [s]")
            plt.ylabel("Membrane Potential")
            plt.show()
        losses.append(loss.item())

    mean_loss = np.mean(losses)
    return losses, mean_loss


def test(model, device, test_loader, epoch, writer=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probabilioty
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100.0 * correct / len(test_loader.dataset)
    logging.info(
        f"\nTest set {FLAGS.model}: Average loss: {test_loss:.4f}, \
            Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n"
    )

    if writer:
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/test", accuracy, epoch)

    return test_loss, accuracy


def save(path, model, optimizer):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load(path, model, optimizer, device):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train(device=device)
    return model, optimizer


def main(args):
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter()
    except ImportError:
        writer = None

    torch.manual_seed(FLAGS.random_seed)

    np.random.seed(FLAGS.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.random_seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(FLAGS.device)

    constant_current_encoder = ConstantCurrentLIFEncoder(
        seq_length=FLAGS.seq_length, p=LIFParameters(v_th=FLAGS.current_encoder_v_th)
    )

    def polar_current_encoder(x):
        x_p = constant_current_encoder(2 * torch.nn.functional.relu(x))
        x_m = constant_current_encoder(2 * torch.nn.functional.relu(-x))
        return torch.cat((x_p, x_m), 1)

    def current_encoder(x):
        x = constant_current_encoder(2 * x)
        return x

    def signed_current_encoder(x):
        z = constant_current_encoder(torch.abs(x))
        return torch.sign(x) * z

    num_channels = 4

    if FLAGS.encoding == "poisson":
        encoder = PoissonEncoder(seq_length=FLAGS.seq_length, f_max=200)
    elif FLAGS.encoding == "constant":
        encoder = current_encoder
    elif FLAGS.encoding == "signed_poisson":
        encoder = SignedPoissonEncoder(seq_length=FLAGS.seq_length, f_max=200)
    elif FLAGS.encoding == "signed_constant":
        encoder = signed_current_encoder
    elif FLAGS.encoding == "constant_polar":
        encoder = polar_current_encoder
        num_channels = 2 * num_channels

    # Load datasets
    transform_norm = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        encoder,
    ]
    transform_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
        ]
        + transform_norm
    )
    transform_test = torchvision.transforms.Compose(transform_norm)
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root=".", train=True, download=True, transform=transform_train
        ),
        batch_size=FLAGS.batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root=".", train=False, transform=transform_test),
        batch_size=FLAGS.batch_size,
        **kwargs,
    )

    label = os.environ.get("SLURM_JOB_ID", str(uuid.uuid4()))
    if not FLAGS.prefix:
        rundir = f"runs/cifar10/{label}"
    else:
        rundir = f"runs/cifar10/{FLAGS.prefix}/{label}"

    os.makedirs(rundir, exist_ok=True)
    os.chdir(rundir)
    FLAGS.append_flags_into_file("flags.txt")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        max_epochs=1000, auto_select_gpus=True, progress_bar_refresh_rate=1
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, help="Number of examples in one minibatch"
    )
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate to use.")
    parser.add_argument(
        "--lr_step",
        type=bool,
        default=True,
        help="Use a stepper to reduce learning weight.",
    )
    parser.add_argument(
        "--current_encoder_v_th",
        type=float,
        default=0.8,
        help="Voltage threshold for the LIF dynamics",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="constant_polar",
        choices=[
            "poisson",
            "constant",
            "constant_first",
            "constant_polar",
            "signed_poisson",
            "signed_constant",
        ],
        help="How to code from CIFAR image to spikes.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["adam", "sgd"],
        help="Optimizer to use for training.",
    )
    parser.add_argument(
        "--seq_length", default=100, type=int, help="Number of timesteps to do."
    )
    parser.add_argument(
        "--manual_seed", default=0, type=int, help="Random seed for torch"
    )
    args = parser.parse_args()

    training_losses = []
    mean_losses = []
    test_losses = []
    accuracies = []

    start = datetime.datetime.now()
    for epoch in range(FLAGS.start_epoch, FLAGS.start_epoch + FLAGS.epochs):
        training_loss, mean_loss = train(
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            lr_scheduler=lr_scheduler,
            writer=writer,
        )
        test_loss, accuracy = test(model, device, test_loader, epoch, writer=writer)

        training_losses += training_loss
        mean_losses.append(mean_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

        if (epoch % FLAGS.model_save_interval == 0) and FLAGS.save_model:
            model_path = f"cifar10-{epoch}.pt"
            save(model_path, model, optimizer)

    stop = datetime.datetime.now()

    np.save("training_losses.npy", np.array(training_losses))
    np.save("mean_losses.npy", np.array(mean_losses))
    np.save("test_losses.npy", np.array(test_losses))
    np.save("accuracies.npy", np.array(accuracies))
    model_path = "cifar10-final.pt"
    save(model_path, model, optimizer)

    logging.info(f"output saved to {rundir}")
    logging.info(f"{start - stop}")
    if writer:
        writer.close()


if __name__ == "__main__":
    app.run(main)
