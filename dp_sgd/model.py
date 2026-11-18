import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn

from .data import PAD

PECSC_REVISION = "bb99df8c344504457555ee1ea93d7b536fc9c32a"


class Receiver(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.channel_decoder = model.channel_decoder
        self.decoder = model.decoder
        self.dense = model.dense

    def forward(self, received, prefix, source_mask):
        memory = self.channel_decoder(received)
        return self.dense(self.decoder(prefix, memory, causal_mask(prefix), source_mask))


def causal_mask(prefix):
    width = prefix.size(1)
    future = torch.ones((width, width), dtype=torch.bool, device=prefix.device).triu(1)
    return (future | prefix.eq(PAD).unsqueeze(1)).float()


class Transmitter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.channel_encoder = model.channel_encoder
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, source, snr, generator):
        valid = source.ne(PAD).unsqueeze(-1)
        source_mask = (~valid).transpose(1, 2).float()
        signal = self.channel_encoder(self.encoder(source, source_mask))
        signal = signal.masked_fill(~valid, 0)
        # Normalize each sentence independently so one record cannot change
        # the received signal or gradient of another record.
        count = valid.sum(dim=1, keepdim=True) * signal.size(-1)
        power = signal.square().sum(dim=(1, 2), keepdim=True) / count.clamp_min(1)
        signal = signal / power.clamp_min(1e-12).sqrt()
        noise_std = (2 * 10 ** (snr / 10)) ** -0.5
        noise = torch.randn(signal.shape, device=signal.device, generator=generator)
        return signal + noise_std * noise, source_mask


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path, record):
    if path.stat().st_size != record["size_bytes"] or sha256(path) != record["sha256"]:
        raise ValueError(f"File does not match the teacher manifest: {path.name}")
    return record["sha256"]


def load_models(args):
    root = args.pecsc.resolve()

    def git(*command):
        return subprocess.check_output(["git", "-C", str(root), *command], text=True).strip()

    if git("rev-parse", "HEAD") != PECSC_REVISION or git(
        "status", "--porcelain", "--untracked-files=all", "--", "models"
    ):
        raise ValueError(f"Use pecSC revision {PECSC_REVISION} with a clean models directory")
    sys.path.insert(0, str(root))
    try:
        from models.transceiver import DeepSC
    finally:
        sys.path.pop(0)
    if Path(sys.modules[DeepSC.__module__].__file__).resolve() != root / "models/transceiver.py":
        raise ValueError("DeepSC was imported from a different checkout")
    if args.synthetic:
        arch = dict(layers=1, vocab_size=16, max_length=6, d_model=8, heads=2, dff=16)
        provenance = {"source": "synthetic random weights"}
    else:
        manifest = json.loads(args.teacher_manifest.read_text(encoding="utf-8"))
        arch = manifest["architecture"]
        selected = manifest["selected_checkpoint"]
        provenance = {
            "teacher_manifest_sha256": sha256(args.teacher_manifest),
            "encoder_sha256": verify(args.encoder, selected["encoder"]),
            "decoder_sha256": verify(args.decoder, selected),
            "tokenizer_sha256": verify(args.tokenizer, manifest["dataset"]["tokenizer_model"]),
            "initialization_provenance": args.initialization_provenance,
        }
    model = DeepSC(
        arch["layers"], arch["vocab_size"], arch["vocab_size"], arch["max_length"],
        arch["max_length"], arch["d_model"], arch["heads"], arch["dff"], dropout=0.0,
    )
    if not args.synthetic:
        encoder = torch.load(args.encoder, map_location="cpu", weights_only=True)
        decoder = torch.load(args.decoder, map_location="cpu", weights_only=True)
        expected_architecture = dict(
            num_layers=arch["layers"], num_heads=arch["heads"], max_len=arch["max_length"],
            d_model=arch["d_model"], dff=arch["dff"], vocab_size=arch["vocab_size"],
        )
        if encoder["meta"] != decoder["meta"] or encoder["meta"]["architecture"] != expected_architecture:
            raise ValueError("The encoder and decoder metadata must match the manifest architecture")
        for name in ("encoder", "channel_encoder"):
            getattr(model, name).load_state_dict(encoder[name], strict=True)
        for name in ("channel_decoder", "decoder", "dense"):
            getattr(model, name).load_state_dict(decoder[name], strict=True)

    return Transmitter(model), Receiver(model), arch, provenance
