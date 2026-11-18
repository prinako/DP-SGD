# DP-SGD

## Differentially Private Decoder Training for Text Semantic Communication

This experiment compares SGD and DP-SGD for text reconstruction in DeepSC across signal-to-noise ratios (SNRs). Both arms start from the same receiver weights, use the same frozen transmitter, and train the channel decoder, semantic decoder, and vocabulary projection for the same update budget.

## Setup

Use Python 3.12 and the pinned pecSC checkout. The pilot imports the existing DeepSC architecture and loads local checkpoints in pecSC's format.

```bash
git clone --branch kd-exp https://github.com/prinako/pecSC.git
git -C pecSC checkout --detach bb99df8c344504457555ee1ea93d7b536fc9c32a
python3.12 -m venv .venv
source .venv/bin/activate
# Optional CPU-only PyTorch wheel
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1
python -m pip install -r requirements.txt
```

Run the offline smoke test and the privacy regression tests.

```bash
python -m dp_sgd --pecsc pecSC --synthetic --output runs/smoke
python -m unittest discover -s tests -v
```

The synthetic run uses a small DeepSC model with random weights, generated sentences, and a token-count cosine metric. It exercises the training, accounting, and evaluation code without research assets or an embedding-model download.

## Research pilot

Use the teacher's `manifest.json`, its selected encoder and decoder checkpoint pair, and the matching SentencePiece BPE model. The CLI verifies their hashes and checks the paired checkpoint metadata. Training and evaluation files accept pecSC token-pair pickles or JSON arrays of `[source_ids, target_ids]` pairs. English reconstruction records start with `<START> <EN>`, end with `<END>`, and have identical source and target IDs. Use a disjoint, shareable held-out split for evaluation.

```bash
python -m dp_sgd --pecsc pecSC --output runs/pilot \
  --teacher-manifest data/teacher/manifest.json \
  --encoder data/teacher/encoder.pth --decoder data/teacher/decoder.pth \
  --tokenizer data/public/tokenizer_bpe.model \
  --train data/private/train_en_en.pkl --evaluation data/public/val_en_en.pkl \
  --initialization-provenance "$ASSET_PROVENANCE" \
  --semantic-revision "$EMBEDDING_MODEL_COMMIT" \
  --epochs 1 --batch-size 4 --train-snr 8 --eval-snrs 2 8 14 \
  --trial-seeds 42 43
```

Set `ASSET_PROVENANCE` to describe how the transmitter, initial receiver, and tokenizer were obtained and their relationship to the protected records. Set `EMBEDDING_MODEL_COMMIT` to the 40-character revision of the specified Sentence Transformers model. The defaults use AWGN, the multilingual MPNet checkpoint used by pecSC's quality benchmark, and CPU. Pass `--device cuda` to use a compatible GPU. Opacus's per-sample gradients can use substantial memory for a full 32K-vocabulary model. Pilot size and batch size can be configured using a prespecified subset and `--batch-size`.

Both trained arms use Poisson sampling, sentence-averaged cross-entropy, and the same learning rate and scheduled number of steps. The pilot applies DeepSC's AWGN noise convention after normalizing each sentence's non-padding channel symbols to unit mean energy. Per-sentence normalization avoids coupling records through a batch-wide normalization factor. At each SNR and trial seed, the initial, SGD, and DP-SGD receivers decode the same noisy batch. Compare scores within this protocol because pecSC's original benchmark uses batch-wide normalization.

## Results and privacy scope

| File | Contents |
| --- | --- |
| `run.json` | Inputs, initial asset fingerprints, training settings, and the achieved RDP-accounted `(epsilon, delta)`. |
| `metrics.json` | chrF++ signature and cosine metric settings. |
| `trials.csv` | Corpus chrF++ and mean sentence embedding cosine by receiver, SNR, and matched noise trial. |
| `summary.csv` | Trial means and paired DP-SGD minus SGD differences at each SNR. |
| `sgd.pt`, `dp_sgd.pt` | Final receiver state dictionaries for the two training arms. |

The accounting covers adding or removing one sentence in a fixed, public roster of training slots. Keep the roster size and sampling schedule fixed across neighboring datasets. Represent an empty training slot as `null` in JSON or `None` in a pecSC pickle and leave that slot in the input. A sentence record is the privacy unit. The default delta is `N^-1.1`, where `N` is the public slot count. DP-SGD uses flat per-record clipping and Gaussian noise. The DP arm draws independent randomness, so reruns can yield different weights and scores. The default RNG is for experimentation. `--secure-mode` requires a compatible `torchcsprng` installation.

The privacy budget is conditional on the fixed initial assets and hyperparameters being independent of the protected records. If a teacher or tokenizer was fitted on those records, the logged budget describes only the decoder fine-tuning mechanism and cannot support an end-to-end privacy claim. Keep the non-private SGD checkpoint and its metrics local when training on protected data. Release only the DP receiver and its metrics from independent, shareable evaluation data under the stated assumptions. Account jointly for multiple DP releases on the same records.
