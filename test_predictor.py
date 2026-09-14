import unittest

import predictor


class PredictorCliTests(unittest.TestCase):
    def test_default_cli_args(self):
        parser = predictor.build_parser()
        args = parser.parse_args([])

        self.assertEqual(args.csv, "AAPL.csv")
        self.assertEqual(args.lags, 5)
        self.assertEqual(args.train_ratio, 0.8)
        self.assertEqual(args.predict_days, 5)
        self.assertTrue(args.plot)

    def test_custom_cli_args(self):
        parser = predictor.build_parser()
        args = parser.parse_args([
            "--csv", "custom.csv",
            "--lags", "10",
            "--train-ratio", "0.7",
            "--predict-days", "3",
            "--no-plot",
        ])

        self.assertEqual(args.csv, "custom.csv")
        self.assertEqual(args.lags, 10)
        self.assertEqual(args.train_ratio, 0.7)
        self.assertEqual(args.predict_days, 3)
        self.assertFalse(args.plot)


if __name__ == "__main__":
    unittest.main()
