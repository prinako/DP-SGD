from copy import deepcopy
import math

import torch
from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import END, EN, PAD, START


def generator(device, seed=None):
    rng = torch.Generator(device=device)
    if seed is None:
        rng.seed()
    else:
        rng.manual_seed(seed)
    return rng


def sentence_losses(logits, expected):
    token_loss = F.cross_entropy(
        logits.transpose(1, 2), expected, reduction="none", ignore_index=PAD
    )
    return token_loss.sum(dim=1) / expected.ne(PAD).sum(dim=1).clamp_min(1)


def train_receiver(initial, transmitter, dataset, args, private):
    receiver = deepcopy(initial).train()
    device = next(receiver.parameters()).device
    optimizer = torch.optim.SGD(receiver.parameters(), lr=args.learning_rate)
    loader = DataLoader(
        dataset.train,
        batch_size=args.batch_size,
        generator=generator("cpu", None if private else args.seed),
    )
    if private:
        engine = PrivacyEngine(accountant="rdp", secure_mode=args.secure_mode)
        receiver, optimizer, loader = engine.make_private(
            module=receiver,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
            loss_reduction="mean",
            poisson_sampling=True,
            noise_generator=None if args.secure_mode else generator(device),
        )
    else:
        loader = DPDataLoader.from_data_loader(loader)

    # Match the expected-batch denominator used by Opacus in both arms.
    expected_batch_size = max(1, int(len(dataset.train) * loader.sample_rate))
    if private:
        optimizer.expected_batch_size = expected_batch_size
    channel_rng = generator(device, None if private else args.seed)
    width = dataset.train.tensors[0].size(1)
    dummy = torch.full((1, width), PAD, dtype=torch.long, device=device)
    dummy[0, :3] = torch.tensor([START, EN, END], device=device)
    for _ in range(args.epochs):
        for (source,) in loader:
            optimizer.zero_grad(set_to_none=True)
            empty = source.size(0) == 0
            source = dummy if empty else source.to(device)
            received, source_mask = transmitter(source, args.train_snr, channel_rng)
            logits = receiver(received, source[:, :-1], source_mask)
            losses = sentence_losses(logits, source[:, 1:])
            # An empty slot or empty draw contributes a public zero gradient.
            # Keep the optimizer step so empty draws still receive noise.
            losses = losses * source[:, 2].ne(END)
            loss = losses.mean() if private else losses.sum() / expected_batch_size
            loss.backward()
            optimizer.step()

    details = {"steps": len(loader) * args.epochs, "sample_rate": loader.sample_rate}
    if private:
        epsilon = float(engine.get_epsilon(args.delta))
        if not math.isfinite(epsilon):
            raise ValueError("The RDP accountant did not return a finite epsilon")
        accounted_steps = sum(steps for _, _, steps in engine.accountant.history)
        if accounted_steps != details["steps"]:
            raise RuntimeError("Privacy accountant did not record every scheduled step")
        details.update(
            epsilon=epsilon, delta=args.delta, accountant="RDP",
            privacy_unit="one sentence record", adjacency="add/remove one record in a fixed public slot roster",
            noise_multiplier=args.noise_multiplier, max_grad_norm=args.max_grad_norm,
            secure_mode=args.secure_mode,
        )
        receiver = receiver.to_standard_module()
    return receiver.eval(), details
