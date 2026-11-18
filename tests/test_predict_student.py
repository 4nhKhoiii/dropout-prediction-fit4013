import tempfile
import unittest

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from dropout_pipeline import (
    EARLY_INPUT_COLS,
    engineer,
    feature_sets,
    make_preprocessor,
    normalize_missing_grades,
    save_best_early_model,
)
from predict_student import load_artifact, parse_input_value, predict_record


def make_record():
    record = {column: 1 for column in EARLY_INPUT_COLS}
    record.update({
        "Previous qualification (grade)": 120.0,
        "Admission grade": 125.0,
        "Age at enrollment": 20.0,
        "Application order": 1.0,
        "Curricular units 1st sem (credited)": 0.0,
        "Curricular units 1st sem (enrolled)": 2.0,
        "Curricular units 1st sem (evaluations)": 2.0,
        "Curricular units 1st sem (approved)": 1.0,
        "Curricular units 1st sem (grade)": 12.0,
        "Curricular units 1st sem (without evaluations)": 0.0,
    })
    return record


class PredictStudentTests(unittest.TestCase):
    def test_best_early_pipeline_round_trips_and_predicts(self):
        num_cols, cat_cols = feature_sets("early", "B")
        records = []
        targets = []
        for index in range(10):
            record = make_record()
            record["Age at enrollment"] = 18.0 + index
            record["Curricular units 1st sem (enrolled)"] = 1.0 + index % 4
            record["Curricular units 1st sem (approved)"] = index % 3
            record["Curricular units 1st sem (grade)"] = 6.0 + index
            records.append(record)
            targets.append(index % 2)

        training = engineer(pd.DataFrame(records), "early")
        features = training[num_cols + cat_cols]
        model = Pipeline([
            ("prep", make_preprocessor(num_cols, cat_cols, scale=False)),
            ("clf", DecisionTreeClassifier(random_state=42)),
        ]).fit(features, targets)
        results = pd.DataFrame([
            {"setting": "early", "flow": "A", "model": "KNN", "cv_f1": 0.70, "best_params": "{}"},
            {"setting": "early", "flow": "B", "model": "DecisionTree", "cv_f1": 0.75, "best_params": "{}"},
            {"setting": "early", "flow": "B", "model": "Baseline_Majority", "cv_f1": 1.0, "best_params": "{}"},
            {"setting": "full", "flow": "B", "model": "DecisionTree", "cv_f1": 1.0, "best_params": "{}"},
        ])
        fitted = {
            (row["setting"], row["flow"], row["model"]):
                (model, features, pd.Series(targets), model.predict(features))
            for row in results.to_dict("records")
        }

        with tempfile.TemporaryDirectory() as model_dir:
            model_path = save_best_early_model(results, fitted, model_dir)
            artifact = load_artifact(model_path)
            probability, prediction = predict_record(records[0], artifact)

        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
        self.assertIn(prediction, (0, 1))
        self.assertEqual(artifact["metadata"]["flow"], "B")
        self.assertEqual(artifact["metadata"]["model_name"], "DecisionTree")

    def test_prediction_rejects_missing_input(self):
        record = make_record()
        artifact = {
            "model": None,
            "metadata": {
                "setting": "early",
                "input_columns": EARLY_INPUT_COLS,
                "feature_columns": [],
            },
        }

        record.pop(EARLY_INPUT_COLS[0])
        with self.assertRaisesRegex(ValueError, "Thiếu trường đầu vào"):
            predict_record(record, artifact)

    def test_binary_input_must_be_zero_or_one(self):
        with self.assertRaisesRegex(ValueError, "chỉ chấp nhận 0 hoặc 1"):
            parse_input_value("Debtor", "2")

    def test_missing_grade_is_imputed_when_there_are_no_evaluations(self):
        record = make_record()
        record["Curricular units 1st sem (evaluations)"] = 0.0
        record["Curricular units 1st sem (grade)"] = 0.0
        normalized = normalize_missing_grades(pd.DataFrame([record]), "early")
        self.assertTrue(pd.isna(normalized.loc[0, "Curricular units 1st sem (grade)"]))
        self.assertTrue(pd.isna(parse_input_value("Admission grade", "")))


if __name__ == "__main__":
    unittest.main()
