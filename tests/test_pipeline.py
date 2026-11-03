"""Small behavioral checks; generated examples are not experimental data."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from he_tgt.history import make_history
from he_tgt.history_models import decision, POLICY
from he_tgt.model import HETGT
from baselines.protocol import build_model


class PipelineChecks(unittest.TestCase):
    def test_history_does_not_use_simultaneous_or_other_stream_rows(self):
        values = np.arange(12, dtype=np.float32).reshape(6, 2)
        times = np.array([0, 10, 10, 11, 41, 42], np.int64) * 10**9
        streams = np.array([0, 0, 0, 1, 0, 0])
        result = make_history(values, times, streams, np.arange(6), np.arange(6))
        np.testing.assert_array_equal(result['history_indices'][1, :2], [0, -1])
        np.testing.assert_array_equal(result['history_indices'][2, :2], [0, -1])
        self.assertTrue((result['history_indices'][3] == -1).all())
        self.assertTrue((result['history_indices'][4] == -1).all())
        np.testing.assert_array_equal(result['history_indices'][5, :2], [4, -1])

    def test_refinement_preserves_mass_and_other_classes(self):
        base = np.array([[.1, .6, .3], [.7, .2, .1]])
        prior = np.array([[.2, .5, .3], [.6, .3, .1]])
        pairs = {(0, 1): np.array([[.3, .4], [.6, .9]]), (1, 2): np.array([[.2, .8], [.4, .2]])}
        mixed = .75 * base + .25 * prior
        final = decision(base, prior, pairs, POLICY)
        np.testing.assert_allclose(final.sum(1), 1)
        self.assertEqual(final[0, 0], mixed[0, 0])
        self.assertEqual(final[1, 2], mixed[1, 2])
        self.assertTrue((final >= 0).all())

    def test_main_graph_supports_training(self):
        torch.set_num_threads(2)
        model = HETGT(12, 8, 3, 6)
        loss = torch.nn.functional.cross_entropy(model(torch.randn(4, 8, 12), torch.eye(8)), torch.tensor([0, 1, 2, 3]))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_all_nine_baselines_support_training(self):
        torch.set_num_threads(2)
        builder = SimpleNamespace(temporal_feat_dim=12, n_nodes=8, n_net=3, n_phy=5)
        for name in ['GCN-AE', 'GAT-AE', 'VGAE', 'GraphSAGE-AE', 'IIoT-GNN', 'EE-GCN', 'STGaAN', 'STCI', 'DT-GNN']:
            with self.subTest(model=name):
                model = build_model(name, builder, 6)
                logits = model(torch.randn(8, 12), torch.eye(8))['anomaly_logits']
                loss = torch.nn.functional.cross_entropy(logits.reshape(1, -1), torch.tensor([2]))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.gnn_encoder.parameters()))


if __name__ == '__main__':
    unittest.main()
