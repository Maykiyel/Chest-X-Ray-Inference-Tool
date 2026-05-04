import tempfile
import unittest
from pathlib import Path
import pandas as pd

from app_services import (
    apply_results_preset,
    audit_folder_quality,
    build_image_consensus_triage,
    build_top_findings_summary,
    create_run_snapshot,
    get_image_explainability,
    list_run_history,
    load_app_config,
    load_snapshot_from_history,
    run_upload_inference,
    save_app_config,
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
        self.assertEqual(list(result.columns), ['filename', 'model', 'top_pathology', 'top_logit'])

    def test_run_upload_inference_batches_and_reports_progress(self):
        files = [DummyUploadFile('a.png', b'a'), DummyUploadFile('b.png', b'b')]
        models = {'nih': object(), 'mimic': object()}
        progress_events = []

        def fake_predict_batch(paths, model, model_name, device, batch_size, auto_label, progress_callback):
            if progress_callback:
                progress_callback(1.0)
            return [{'model': model_name, 'filename': paths[0].name, 'pathology': 'A', 'probability': 0.8}]

        results = run_upload_inference(
            files,
            models,
            device='cpu',
            batch_size=99,
            progress_callback=lambda idx, p: progress_events.append((idx, p)),
            predict_batch_fn=fake_predict_batch,
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(progress_events, [(0, 1.0), (1, 1.0)])

    def test_run_upload_inference_forwards_auto_label_flag(self):
        files = [DummyUploadFile('Atelectasis_1.jpg', b'a')]
        models = {'nih': object()}
        captured = {'auto_label': None}

        def fake_predict_batch(paths, model, model_name, device, batch_size, auto_label, progress_callback):
            captured['auto_label'] = auto_label
            return [{'model': model_name, 'filename': paths[0].name, 'pathology': 'Atelectasis', 'probability': 0.9}]

        run_upload_inference(
            files,
            models,
            device='cpu',
            batch_size=1,
            auto_label=True,
            predict_batch_fn=fake_predict_batch,
        )

        self.assertTrue(captured['auto_label'])

    def test_apply_results_preset_high_confidence(self):
        df = pd.DataFrame([
            {'filename': 'a', 'model': 'nih', 'pathology': 'A', 'probability': 0.65},
            {'filename': 'b', 'model': 'nih', 'pathology': 'B', 'probability': 0.91},
        ])
        filtered = apply_results_preset(df, 'High logit (≥0.847)')
        self.assertEqual(len(filtered), 1)

    def test_create_run_snapshot_and_history_roundtrip(self):
        df = pd.DataFrame([{'filename': 'x', 'model': 'nih', 'pathology': 'A', 'probability': 0.8}])
        snapshot = create_run_snapshot(df, {'mode': 'upload'}, 'History test')

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_snapshot_to_history(snapshot, base_dir=tmp_dir)
            records = list_run_history(base_dir=tmp_dir)
            self.assertEqual(len(records), 1)
            loaded = load_snapshot_from_history(snapshot['id'], base_dir=tmp_dir)
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded['results']), 1)

    def test_config_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            from app_services import CONFIG_FILE
            original = CONFIG_FILE
            try:
                import app_services
                app_services.CONFIG_FILE = Path(tmp_dir) / '.app_config.json'
                save_app_config({'default_batch_size': 8})
                loaded = load_app_config()
                self.assertEqual(loaded.get('default_batch_size'), 8)
            finally:
                app_services.CONFIG_FILE = original

    def test_audit_and_explainability(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_path = Path(tmp_dir) / 'Atelectasis_img1.png'
            from PIL import Image
            Image.new('L', (300, 300), color=128).save(img_path)
            stats = audit_folder_quality(Path(tmp_dir), recursive=False)
            self.assertEqual(stats['total_images'], 1)
            self.assertGreaterEqual(stats['label_percentage'], 0)

        df = pd.DataFrame([
            {'filename': 'x.png', 'model': 'nih', 'pathology': 'A', 'probability': 0.9},
            {'filename': 'x.png', 'model': 'nih', 'pathology': 'Normal', 'probability': 0.1},
            {'filename': 'x.png', 'model': 'mimic', 'pathology': 'A', 'probability': 0.8},
            {'filename': 'x.png', 'model': 'mimic', 'pathology': 'Normal', 'probability': 0.2},
        ])
        explain = get_image_explainability(df, 'x.png')
        self.assertTrue(explain['available'])
        self.assertIn('top3', explain)

    def test_build_image_consensus_triage_assigns_risk_bands(self):
        df = pd.DataFrame([
            {'filename': 'a.png', 'model': 'nih', 'pathology': 'Pneumonia', 'probability': 0.91},
            {'filename': 'a.png', 'model': 'mimic', 'pathology': 'Pneumonia', 'probability': 0.88},
            {'filename': 'a.png', 'model': 'chexpert', 'pathology': 'Pneumonia', 'probability': 0.84},
            {'filename': 'b.png', 'model': 'nih', 'pathology': 'Pneumonia', 'probability': 0.62},
            {'filename': 'b.png', 'model': 'mimic', 'pathology': 'Pneumonia', 'probability': 0.53},
            {'filename': 'b.png', 'model': 'chexpert', 'pathology': 'Pneumonia', 'probability': 0.40},
            {'filename': 'c.png', 'model': 'nih', 'pathology': 'Pneumonia', 'probability': 0.22},
            {'filename': 'c.png', 'model': 'mimic', 'pathology': 'Pneumonia', 'probability': 0.19},
            {'filename': 'c.png', 'model': 'chexpert', 'pathology': 'Pneumonia', 'probability': 0.15},
        ])

        triage = build_image_consensus_triage(df, positive_threshold=0.5, high_risk_threshold=0.75)
        self.assertEqual(len(triage), 3)

        risk_by_file = {row['filename']: row['risk_band'] for _, row in triage.iterrows()}
        self.assertEqual(risk_by_file['a.png'], 'High')
        self.assertEqual(risk_by_file['b.png'], 'Moderate')
        self.assertEqual(risk_by_file['c.png'], 'Low')

    def test_build_image_consensus_triage_empty_schema(self):
        triage = build_image_consensus_triage(pd.DataFrame())
        self.assertEqual(
            list(triage.columns),
            [
                'filename',
                'pathology',
                'models_reporting',
                'mean_probability',
                'max_probability',
                'std_probability',
                'positive_votes',
                'vote_fraction',
                'risk_band',
            ],
        )


if __name__ == '__main__':
    unittest.main()
