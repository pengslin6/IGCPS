"""Check reference markings and deterministic ranking in TXT reports."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import export_traceback_reports as report
import trace as tr


class ExportReportTests(unittest.TestCase):
    def test_ns_heater_is_not_marked_as_reference(self):
        columns = [tr.IGCPS_NS_ROOT_COLUMNS[0],
                   'phy_Switch status of medium pressure skid heater',
                   tr.IGCPS_NS_ROOT_COLUMNS[1]]
        builder = SimpleNamespace(phy_cols=columns, phy_nodes=columns, n_net=1, n_phy=3,
                                  ip_to_idx={'10.0.6.118': 0}, unique_ips=['10.0.6.118'])
        data = {'y': np.array([1]), 'label_names': ['Normal', 'NS'],
                'src_ips': ['10.0.6.118'], 'dst_ips': ['10.0.6.118']}
        scores = np.array([[.0, .8, .9, .8]])
        metrics = {'confusion_matrix': [[0, 0], [0, 1]], 'accuracy': 100., 'f1': 100.,
                   'auc': 100., 'top1': 0., 'mrr': 50., 'ndcg5': 60.,
                   'mean_reference_rank': 2., 'trace_eval_total': 1, 'trace_eval_skipped': 0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.txt'
            report.write_report(path, 'IGCPS', 'test fixture', data, builder,
                                np.array([[.01, .99]]), scores, metrics, Path('fixture.pt'))
            text = path.read_text(encoding='utf-8')
        reference_line = next(line for line in text.splitlines() if line.startswith('Reference node(s):'))
        self.assertNotIn('heater', reference_line)
        heater_line = next(line for line in text.splitlines() if '1. Process:' in line)
        self.assertIn('heater', heater_line)
        self.assertNotIn('[REFERENCE]', heater_line)
        self.assertIn('First reference rank: 2', text)
        self.assertIn('2. Process:' + tr.IGCPS_NS_ROOT_COLUMNS[1], text)
        self.assertNotIn('Checkpoint supervision: legacy NS three-switch references', text)
        self.assertIn('not the five-seed mean', text)

    def test_legacy_igcps_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.pt'
            torch.save({'state_dict': {}}, path)
            with self.assertRaises(ValueError):
                report.validate_checkpoint_policy(path, 'IGCPS')

    def test_corrected_igcps_policy_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.pt'
            torch.save({'root_reference_policy': tr.ROOT_REFERENCE_POLICY}, path)
            result = report.validate_checkpoint_policy(path, 'IGCPS')
            self.assertEqual(result['checkpoint_reference_policy'], tr.ROOT_REFERENCE_POLICY)

    def test_metric_mismatch_still_raises(self):
        with self.assertRaises(RuntimeError):
            report.validate_metrics('fixture', {'accuracy': 80.}, {'accuracy': 90.})


if __name__ == '__main__':
    unittest.main()
