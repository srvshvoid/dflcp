from __future__ import annotations

import unittest

from fedforge.distributed import UpdateEnvelope, aggregate_masked
from fedforge.security import clip_and_noise, mask_update, unmask_sum


class SecurityProtocolTests(unittest.TestCase):
    def test_pairwise_masks_cancel(self) -> None:
        clients = ("client-a", "client-b", "client-c")
        secret = "local-test-secret"
        values = {
            client: tuple(float(index + 1) for index in range(4))
            for client in clients
        }
        masked = [
            mask_update(client, values[client], clients, secret)
            for client in clients
        ]
        recovered = unmask_sum(masked, clients, secret)
        expected = tuple(sum(values[client][index] for client in clients) for index in range(4))
        for actual, target in zip(recovered, expected):
            self.assertAlmostEqual(actual, target, places=12)

    def test_signed_masked_envelopes_aggregate(self) -> None:
        clients = ("a", "b")
        secret = "secret"
        envelopes = [
            UpdateEnvelope.create(client, 1, (1.0, 2.0), 10, secret, peers=clients, secure_aggregation=True)
            for client in clients
        ]
        self.assertEqual(aggregate_masked(envelopes, clients, secret), (2.0, 4.0))
        tampered = UpdateEnvelope(
            envelopes[0].client_id, envelopes[0].round_id, (99.0, 2.0),
            envelopes[0].num_examples, envelopes[0].masked, envelopes[0].signature,
        )
        with self.assertRaises(ValueError):
            aggregate_masked([tampered, envelopes[1]], clients, secret)

    def test_dp_clips_update(self) -> None:
        result = clip_and_noise((3.0, 4.0), (0.0, 0.0), clip_norm=1.0)
        self.assertAlmostEqual(sum(value * value for value in result) ** 0.5, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
