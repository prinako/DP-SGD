import argparse
import os
from pathlib import Path
import unittest

import torch
from opacus.grad_sample import GradSampleModule

from dp_sgd.data import synthetic_data
from dp_sgd.model import load_models
from dp_sgd.training import generator, sentence_losses, train_receiver


class PrivacyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        args = argparse.Namespace(
            synthetic=True,
            pecsc=Path(os.environ.get("PECSC_DIR", "pecSC")),
        )
        self.transmitter, self.receiver, _, _ = load_models(args)
        self.dataset = synthetic_data()

    def test_record_gradient_is_independent_of_batch_peer(self):
        rows = self.dataset.train.tensors[0][:2]
        receiver = GradSampleModule(self.receiver)
        batch_received, batch_mask = self.transmitter(rows, 8, generator("cpu", 7))
        batch = receiver(batch_received, rows[:, :-1], batch_mask)
        sentence_losses(batch, rows[:, 1:]).mean().backward()
        batch_gradients = [p.grad_sample[0].clone() for p in receiver.parameters()]
        receiver.zero_grad(set_to_none=True)

        different_peer = torch.stack((rows[0], self.dataset.train.tensors[0][-1]))
        received, mask = self.transmitter(different_peer, 8, generator("cpu", 7))
        torch.testing.assert_close(batch_received[0], received[0])
        single = receiver(received[:1], rows[:1, :-1], mask[:1])
        sentence_losses(single, rows[:1, 1:]).mean().backward()
        for expected, parameter in zip(batch_gradients, receiver.parameters()):
            torch.testing.assert_close(expected, parameter.grad_sample[0], atol=1e-6, rtol=1e-4)

    def test_private_training_preserves_initial_weights_and_counts_steps(self):
        args = argparse.Namespace(
            batch_size=1, epochs=1, seed=42, learning_rate=0.01, noise_multiplier=1.0,
            max_grad_norm=1.0, delta=0.001, secure_mode=False, train_snr=8.0,
        )
        before = {name: tensor.clone() for name, tensor in self.receiver.state_dict().items()}
        trained, privacy = train_receiver(self.receiver, self.transmitter, self.dataset, args, True)
        self.assertEqual(privacy["steps"], len(self.dataset.train))
        self.assertGreater(privacy["epsilon"], 0)
        self.assertTrue(any(not torch.equal(trained.state_dict()[name], tensor)
                            for name, tensor in before.items()))
        for name, tensor in self.receiver.state_dict().items():
            torch.testing.assert_close(tensor, before[name])
        self.assertTrue(all(parameter.grad is None for parameter in self.transmitter.parameters()))


if __name__ == "__main__":
    unittest.main()
