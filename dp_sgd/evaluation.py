from collections import Counter
from importlib.metadata import version

import torch
from sacrebleu.metrics import CHRF
from torch.utils.data import DataLoader

from .data import EN, END, PAD, START
from .model import causal_mask
from .training import generator


def greedy_decode(receiver, received, mask, width):
    prefix = received.new_tensor([START, EN], dtype=torch.long).repeat(received.size(0), 1)
    finished = torch.zeros(prefix.size(0), device=prefix.device, dtype=torch.bool)
    memory = receiver.channel_decoder(received)
    for _ in range(width - 2):
        states = receiver.decoder(prefix, memory, causal_mask(prefix), mask)
        token = receiver.dense(states[:, -1]).argmax(dim=-1)
        token = token.masked_fill(finished, PAD)
        prefix = torch.cat((prefix, token.unsqueeze(1)), dim=1)
        finished |= token.eq(END)
        if finished.all():
            break
    return prefix.tolist()


def synthetic_embeddings(texts, vocabulary):
    rows = [Counter(text.split()) for text in texts]
    return torch.tensor([[counts[word] for word in vocabulary] for counts in rows],
                        dtype=torch.float32)


def embedding_encoder(args):
    if args.synthetic:
        words = "send receive signal message noise channel text data <unk>".split()
        return (lambda texts: synthetic_embeddings(texts, words)), {
            "metric": "synthetic_token_count_cosine",
        }

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        args.semantic_model, revision=args.semantic_revision, device=args.device
    )
    model.eval()

    def encode(texts):
        return model.encode(texts, convert_to_tensor=True, show_progress_bar=False,
                            batch_size=args.embedding_batch_size)

    return encode, {
        "metric": "embedding_cosine", "model": args.semantic_model,
        "revision": args.semantic_revision, "library_version": version("sentence-transformers"),
    }


@torch.inference_mode()
def evaluate(transmitter, models, dataset, args):
    loader = DataLoader(dataset.evaluation, batch_size=args.eval_batch_size)
    width = dataset.evaluation.tensors[0].size(1)
    references = [dataset.decode(row) for row in dataset.evaluation.tensors[0].tolist()]
    encode, embedding_metadata = embedding_encoder(args)
    reference_vectors = encode(references)
    chrf = CHRF(word_order=2)
    rows = []
    for snr in args.eval_snrs:
        for trial in args.trial_seeds:
            rng = generator(args.device, trial)
            predictions = {name: [] for name in models}
            for (source,) in loader:
                received, mask = transmitter(source.to(args.device), snr, rng)
                for name, receiver in models.items():
                    decoded = greedy_decode(receiver, received, mask, width)
                    predictions[name].extend(dataset.decode(row) for row in decoded)
            for name, texts in predictions.items():
                cosine = torch.nn.functional.cosine_similarity(reference_vectors, encode(texts))
                rows.append({
                    "model": name, "snr_db": snr, "trial_seed": trial,
                    "sentence_count": len(references),
                    "chrfpp": chrf.corpus_score(texts, [references]).score,
                    embedding_metadata["metric"]: cosine.mean().item(),
                })
    return rows, {"chrfpp": str(chrf.get_signature()), "cosine": embedding_metadata}
