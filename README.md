# DP-SGD

## Differentially Private Decoder Training for Text Semantic Communication: An Evaluation Across SNR Levels

This practical study compares standard SGD and differentially private SGD (DP-SGD) for training a decoder in a text semantic communication system. It asks how the difference in text reconstruction quality between the two training methods varies with the channel signal-to-noise ratio (SNR).

## Planned experiment

- Freeze the transmitter and train two decoders with the same architecture and initial weights.
- Use the same training split and epoch budget for both methods. Use Opacus to apply DP-SGD and record the achieved privacy budget (ε, δ) and the unit of privacy.
- Evaluate both decoders on the same held-out sentences and matched channel noise realizations. Start with one channel and three SNR levels before running the full sweep.
- Report chrF++ and embedding cosine similarity by SNR, alongside the privacy budget and the assumptions about the provenance of the frozen transmitter and initial decoder weights.

## Status

The experiment is planned. The first milestone is to reproduce the existing evaluation and complete a short training run for each method.
