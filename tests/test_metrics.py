import unittest

from metrics import compute_confusion_matrix_metrics, recommend_threshold


class MetricsThresholdRecommendationTests(unittest.TestCase):
    def test_compute_confusion_matrix_metrics_all_positive_is_mapped_correctly(self):
        y_true = [1, 1, 1]
        y_pred = [0.9, 0.8, 0.95]

        result = compute_confusion_matrix_metrics(y_true, y_pred, threshold=0.5)

        self.assertEqual(result['confusion_matrix'].shape, (2, 2))
        self.assertEqual(result['true_positives'], 3)
        self.assertEqual(result['true_negatives'], 0)
        self.assertEqual(result['false_positives'], 0)
        self.assertEqual(result['false_negatives'], 0)

    def test_recommend_threshold_returns_metrics_payload(self):
        y_true = [0, 0, 0, 1, 1, 1]
        y_pred = [0.05, 0.15, 0.30, 0.70, 0.85, 0.95]

        result = recommend_threshold(y_true, y_pred, strategy='youden')

        self.assertIn('threshold', result)
        self.assertIn('score', result)
        self.assertIn('metrics', result)
        self.assertGreaterEqual(result['threshold'], 0.05)
        self.assertLessEqual(result['threshold'], 0.95)

    def test_recommend_threshold_supports_f1(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0.1, 0.4, 0.6, 0.9]

        result = recommend_threshold(y_true, y_pred, strategy='f1')
        self.assertEqual(result['strategy'], 'f1')
        self.assertGreaterEqual(result['score'], 0.0)


if __name__ == '__main__':
    unittest.main()
