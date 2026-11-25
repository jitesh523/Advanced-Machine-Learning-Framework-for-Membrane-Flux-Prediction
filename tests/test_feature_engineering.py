"""Regression tests for category encoding across training and prediction batches."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.linear_model import LinearRegression

from src.feature_engineering import FeatureEngineer


ONEHOT_COLS = [
    'Membrane AL Type',
    'Membrane Direction',
    'Flow Orientation (counter/ cocurrent)',
]
TARGET = 'Membrane Flux (LMH)'


def training_data():
    data = {col: ['A', 'B', 'C'] for col in ONEHOT_COLS}
    data.update({
        'DS Osmotic pressure (atm)': [10.0, 20.0, 30.0],
        'FS Osmotic pressure (atm)': [1.0, 2.0, 3.0],
        'DS Temperature (°C)': [20.0, 25.0, 30.0],
        'FS Temperature (°C)': [20.0, 20.0, 20.0],
        'DS Velocity (cm/s)': [1.0, 2.0, 3.0],
        'FS Velocity (cm/s)': [1.0, 1.0, 1.0],
        'DS Molarity (M)': [0.1, 0.2, 0.3],
        'FS Molarity (M)': [0.01, 0.02, 0.03],
        'DS Molecular weight': [58.44, 58.44, 58.44],
        'Manufacture': ['X', 'Y', 'X'],
        'Feed Solution (FS)': ['NaCl', 'DI', 'NaCl'],
        'Draw Solution (DS)': ['NaCl', 'NaCl', 'NaCl'],
        TARGET: [5.0, 10.0, 15.0],
    })
    return pd.DataFrame(data, index=[10, 20, 30])


class FeatureEngineeringTests(unittest.TestCase):
    def test_single_rows_match_training_encoding_and_predictions(self):
        raw = training_data()
        engineer = FeatureEngineer()
        trained = engineer.fit_transform(raw)
        model = LinearRegression().fit(trained.drop(columns=TARGET), raw[TARGET])
        for index in raw.index:
            with self.subTest(index=index):
                actual = engineer.transform(raw.loc[[index]].drop(columns=TARGET))
                expected = trained.loc[[index]].drop(columns=TARGET)
                assert_frame_equal(actual, expected)
                np.testing.assert_allclose(model.predict(actual), model.predict(expected))

    def test_subset_batch_keeps_training_reference_category(self):
        engineer = FeatureEngineer()
        raw = training_data()
        trained = engineer.encode_categorical_features(raw)
        actual = engineer.encode_categorical_features(raw.iloc[1:], fit=False)
        assert_frame_equal(actual, trained.iloc[1:])
        for col in ONEHOT_COLS:
            self.assertTrue(actual.loc[20, f'{col}_B'])
            self.assertFalse(actual.loc[20, f'{col}_C'])

    def test_unknown_categories_do_not_add_columns_or_shift_known_values(self):
        engineer = FeatureEngineer()
        raw = training_data()
        trained = engineer.encode_categorical_features(raw)
        prediction = raw.iloc[1:].copy()
        for col in ONEHOT_COLS:
            prediction.loc[30, col] = '0-unseen'
        actual = engineer.encode_categorical_features(prediction, fit=False)
        self.assertEqual(list(actual.columns), list(trained.columns))
        assert_frame_equal(actual.loc[[20]], trained.loc[[20]])
        for col in ONEHOT_COLS:
            self.assertFalse(actual.loc[30, f'{col}_B'])
            self.assertFalse(actual.loc[30, f'{col}_C'])

    def test_refitting_replaces_category_schema(self):
        engineer = FeatureEngineer()
        raw = training_data()
        engineer.encode_categorical_features(raw)
        for col in ONEHOT_COLS:
            raw[col] = ['D', 'E', 'F']
        trained = engineer.encode_categorical_features(raw)
        actual = engineer.encode_categorical_features(raw.iloc[[1]], fit=False)
        assert_frame_equal(actual, trained.iloc[[1]])

    def test_saved_pipeline_retains_category_schema(self):
        engineer = FeatureEngineer()
        raw = training_data()
        trained = engineer.fit_transform(raw)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'engineer.pkl'
            engineer.save(path)
            loaded = FeatureEngineer.load(path)
            assert_frame_equal(loaded.transform(raw.iloc[[1]]), trained.iloc[[1]])

    def test_legacy_pipeline_uses_stored_feature_names(self):
        engineer = FeatureEngineer()
        raw = training_data()
        trained = engineer.fit_transform(raw)
        # Simulate the state persisted by the previous implementation.
        del engineer.onehot_columns
        assert_frame_equal(engineer.transform(raw.iloc[[1]]), trained.iloc[[1]])
        assert_frame_equal(engineer.fit_transform(raw), trained)


if __name__ == '__main__':
    unittest.main()
