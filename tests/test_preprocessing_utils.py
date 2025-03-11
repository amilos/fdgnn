import pandas as pd
import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler
from scripts import preprocess_utils

# filepath: /Users/aca/UoL/final/fdgnn/tests/test_preprocessing_utils.py

class TestPreprocessingUtils(unittest.TestCase):

    def test_create_device_profile_id_less_than_3_nan(self):
        row = pd.Series({
            "id_30": "Mozilla",
            "id_31": "chrome",
            "id_32": "32",
            "id_33": "1920x1080",
            "DeviceType": "desktop",
            "DeviceInfo": "windows"
        })
        expected = "Mozilla_chrome_32_1920x1080_desktop_windows"
        actual = preprocess_utils.create_device_profile_id(row)
        self.assertEqual(actual, expected)

    def test_create_device_profile_id_more_than_3_nan(self):
        row = pd.Series({
            "id_30": "nan",
            "id_31": "chrome",
            "id_32": "nan",
            "id_33": "nan",
            "DeviceType": "desktop",
            "DeviceInfo": "windows"
        })
        expected = "missing_device"
        actual = preprocess_utils.create_device_profile_id(row)
        self.assertEqual(actual, expected)

    def test_create_device_profile_id_exactly_3_nan(self):
        row = pd.Series({
            "id_30": "Mozilla",
            "id_31": "nan",
            "id_32": "nan",
            "id_33": "nan",
            "DeviceType": "desktop",
            "DeviceInfo": "windows"
        })
        expected = "missing_device"
        actual = preprocess_utils.create_device_profile_id(row)
        self.assertEqual(actual, expected)

    def test_create_device_profile_id_all_not_nan(self):
        row = pd.Series({
            "id_30": "Mozilla",
            "id_31": "chrome",
            "id_32": "32",
            "id_33": "1920x1080",
            "DeviceType": "desktop",
            "DeviceInfo": "windows"
        })
        expected = "Mozilla_chrome_32_1920x1080_desktop_windows"
        actual = preprocess_utils.create_device_profile_id(row)
        self.assertEqual(actual, expected)

    def test_create_device_profile_id_all_nan(self):
        row = pd.Series({
            "id_30": None,
            "id_31": None,
            "id_32": None,
            "id_33": None,
            "DeviceType": None,
            "DeviceInfo": None
        })
        expected = "missing_device"
        actual = preprocess_utils.create_device_profile_id(row)
        self.assertEqual(actual, expected)

    def test_create_device_profile_id_empty_strings(self):
        row = pd.Series({
            "id_30": "",
            "id_31": "",
            "id_32": "",
            "id_33": "",
            "DeviceType": "",
            "DeviceInfo": ""
        })
        expected = "___desktop_"
        actual = preprocess_utils.create_device_profile_id(row)
        self.assertEqual(actual, '____desktop_')

    def test_create_device_profile_id_mixed_nan_and_values(self):
        row = pd.Series({
            "id_30": "Mozilla",
            "id_31": None,
            "id_32": "32",
            "id_33": None,
            "DeviceType": "desktop",
            "DeviceInfo": None
        })
        expected = "Mozilla_nan_32_nan_desktop_nan"
        actual = preprocess_utils.create_device_profile_id(row)
        self.assertEqual(actual, expected)

    def test_scale_numeric_features_fit(self):
        df = pd.DataFrame({'col1': [1.0, 2.0, 3.0], 'col2': [4.0, 5.0, 6.0]})
        num_cols = ['col1', 'col2']
        df_scaled, scaler = preprocess_utils.scale_numeric_features(df, num_cols)

        self.assertIsInstance(df_scaled, pd.DataFrame)
        self.assertIsInstance(scaler, StandardScaler)
        self.assertEqual(df_scaled.shape, df.shape)
        self.assertTrue(np.allclose(df_scaled.mean().values, 0))
        self.assertTrue(np.allclose(df_scaled.std().values, 1))

    def test_scale_numeric_features_transform(self):
        df_train = pd.DataFrame({'col1': [1.0, 2.0, 3.0], 'col2': [4.0, 5.0, 6.0]})
        num_cols = ['col1', 'col2']
        df_train_scaled, scaler = preprocess_utils.scale_numeric_features(df_train, num_cols)

        df_test = pd.DataFrame({'col1': [4.0, 5.0, 6.0], 'col2': [7.0, 8.0, 9.0]})
        df_test_scaled, _ = preprocess_utils.scale_numeric_features(df_test, num_cols, scaler=scaler, fit=False)

        self.assertIsInstance(df_test_scaled, pd.DataFrame)
        self.assertEqual(df_test_scaled.shape, df_test.shape)

        # Verify that the test data is transformed using the training scaler
        scaler_check = StandardScaler()
        scaler_check.fit(df_train)
        expected_scaled = scaler_check.transform(df_test)
        self.assertTrue(np.allclose(df_test_scaled.values,expected_scaled))

    def test_scale_numeric_features_missing_columns(self):
        df = pd.DataFrame({'col1': [1.0, 2.0, 3.0]})
        num_cols = ['col1', 'col2']
        with self.assertRaises(KeyError):
            preprocess_utils.scale_numeric_features(df, num_cols)

    def test_scale_numeric_features_no_scaler_provided(self):
        df = pd.DataFrame({'col1': [1.0, 2.0, 3.0]})
        num_cols = ['col1']
        with self.assertRaises(ValueError):
            preprocess_utils.scale_numeric_features(df, num_cols, fit=False)

    def test_scale_numeric_features_empty_dataframe(self):
        df = pd.DataFrame()
        num_cols = []
        df_scaled, scaler = preprocess_utils.scale_numeric_features(df, num_cols)
        self.assertTrue(df_scaled.empty)
        self.assertIsInstance(scaler, StandardScaler)

    def test_scale_numeric_features_already_fitted_scaler(self):
        df = pd.DataFrame({'col1': [1.0, 2.0, 3.0], 'col2': [4.0, 5.0, 6.0]})
        num_cols = ['col1', 'col2']
        scaler = StandardScaler()
        scaler.fit(df[num_cols])
        df_scaled, scaler_returned = preprocess_utils.scale_numeric_features(df, num_cols, scaler=scaler, fit=False)

        self.assertIsInstance(df_scaled, pd.DataFrame)
        self.assertIsInstance(scaler_returned, StandardScaler)
        self.assertEqual(df_scaled.shape, df.shape)
        self.assertTrue(np.allclose(df_scaled.values, scaler.transform(df[num_cols]).data))

    def test_scale_numeric_features_columns_mismatch(self):
        df_train = pd.DataFrame({'col1': [1.0, 2.0, 3.0], 'col2': [4.0, 5.0, 6.0]})
        num_cols = ['col1', 'col2']
        df_train_scaled, scaler = preprocess_utils.scale_numeric_features(df_train, num_cols)

        df_test = pd.DataFrame({'col1': [4.0, 5.0, 6.0]})
        num_cols_test = ['col1', 'col3']
        with self.assertRaises(KeyError):
            preprocess_utils.scale_numeric_features(df_test, num_cols_test, scaler=scaler, fit=False)        

if __name__ == '__main__':
    unittest.main()