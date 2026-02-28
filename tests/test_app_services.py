import tempfile
import unittest
import pandas as pd

from app_services import (
    apply_results_preset,
    build_top_findings_summary,
    create_run_snapshot,
    list_run_history,
    load_snapshot_from_history,
    run_upload_inference,
    save_snapshot_to_history,
)


class DummyUploadFile:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getbuffer(self):
        return self._data


class AppServicesTests(unittest.TestCase):
    def test_build_top_findings_summary_returns_stable_empty_schema(self):
        result = build_top_findings_summary(pd.DataFrame())
        self.assertEqual(
            list(result.columns),
            ['filename', 'model', 'top_pathology', 'top_probability'],
        )

    def test_run_upload_inference_batches_and_reports_progress(self):
        files = [DummyUploadFile('a.png', b'a'), DummyUploadFile('b.png', b'b')]
        models = {'nih': object(), 'mimic': object()}

        call_log = []
        progress_events = []

        def fake_predict_batch(paths, model, model_name, device, batch_size, auto_label, progress_callback):
            call_log.append({
                'model_name': model_name,
                'device': device,
                'batch_size': batch_size,
                'auto_label': auto_label,
                'path_count': len(paths),
            })
            if progress_callback:
                progress_callback(0.5)
                progress_callback(1.0)
            return [{'model': model_name, 'filename': paths[0].name}]

        results = run_upload_inference(
            files,
            models,
            device='cpu',
            batch_size=99,
            progress_callback=lambda idx, p: progress_events.append((idx, p)),
            predict_batch_fn=fake_predict_batch,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual([item['model'] for item in results], ['nih', 'mimic'])
        self.assertEqual([entry['batch_size'] for entry in call_log], [2, 2])
        self.assertEqual([entry['auto_label'] for entry in call_log], [False, False])
        self.assertEqual(progress_events, [(0, 0.5), (0, 1.0), (1, 0.5), (1, 1.0)])

    def test_apply_results_preset_high_confidence(self):
        df = pd.DataFrame([
            {'filename': 'a', 'model': 'nih', 'pathology': 'A', 'probability': 0.65},
            {'filename': 'b', 'model': 'nih', 'pathology': 'B', 'probability': 0.91},
        ])
        filtered = apply_results_preset(df, 'High confidence (≥0.70)')
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]['pathology'], 'B')

    def test_create_run_snapshot_includes_expected_fields(self):
        df = pd.DataFrame([{'filename': 'a', 'model': 'nih', 'pathology': 'A', 'probability': 0.7}])
        snapshot = create_run_snapshot(df, {'mode': 'upload'}, 'Quick run')
        self.assertIn('id', snapshot)
        self.assertEqual(snapshot['label'], 'Quick run')
        self.assertEqual(snapshot['rows'], 1)
        self.assertEqual(snapshot['stats']['mode'], 'upload')
        self.assertTrue(isinstance(snapshot['results'], pd.DataFrame))

    def test_persisted_history_roundtrip(self):
        df = pd.DataFrame([{'filename': 'x', 'model': 'nih', 'pathology': 'A', 'probability': 0.8}])
        snapshot = create_run_snapshot(df, {'mode': 'upload'}, 'History test')

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_snapshot_to_history(snapshot, base_dir=tmp_dir)
            records = list_run_history(base_dir=tmp_dir)
            self.assertEqual(len(records), 1)
            loaded = load_snapshot_from_history(snapshot['id'], base_dir=tmp_dir)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded['label'], 'History test')
            self.assertEqual(len(loaded['results']), 1)


if __name__ == '__main__':
    unittest.main()
