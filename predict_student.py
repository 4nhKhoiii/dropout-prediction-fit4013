import argparse
import math

import joblib
import pandas as pd

from dropout_pipeline import BIN_COLS, CAT_COLS, engineer, normalize_missing_grades


def parse_input_value(column, value):
    """Parse and validate a numeric dataset value entered in the terminal."""
    if value is None or str(value).strip().casefold() in {"", "na", "nan"}:
        return float("nan")

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column}: vui lòng nhập một giá trị số.") from exc

    if not math.isfinite(number):
        raise ValueError(f"{column}: giá trị phải là số hữu hạn.")

    if column in CAT_COLS:
        if not number.is_integer():
            raise ValueError(f"{column}: mã danh mục phải là số nguyên.")
        return int(number)

    if column in BIN_COLS:
        if number not in (0, 1):
            raise ValueError(f"{column}: chỉ chấp nhận 0 hoặc 1.")
        return int(number)

    return number


def collect_input(input_func=input, output_func=print, input_columns=None):
    """Prompt for all raw fields required by the saved model."""
    if input_columns is None:
        raise ValueError("Thiếu danh sách cột đầu vào trong metadata mô hình.")

    record = {}
    for column in input_columns:
        while True:
            value = input_func(f"{column}: ")
            try:
                record[column] = parse_input_value(column, value)
                break
            except ValueError as exc:
                output_func(str(exc))
    return record


def load_artifact(path):
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or not {"model", "metadata"} <= artifact.keys():
        raise ValueError("File mô hình không đúng định dạng artifact dự kiến.")

    metadata = artifact["metadata"]
    required_metadata = {"setting", "input_columns", "feature_columns"}
    if not isinstance(metadata, dict) or not required_metadata <= metadata.keys():
        raise ValueError("Artifact thiếu metadata cần thiết để dự đoán.")
    if metadata["setting"] != "early":
        raise ValueError("Script này chỉ hỗ trợ mô hình phạm vi early.")
    return artifact


def predict_record(record, artifact):
    metadata = artifact["metadata"]
    input_columns = metadata["input_columns"]
    missing = [column for column in input_columns if column not in record]
    if missing:
        raise ValueError("Thiếu trường đầu vào: " + ", ".join(missing))

    validated = {
        column: parse_input_value(column, record[column])
        for column in input_columns
    }
    data = normalize_missing_grades(
        pd.DataFrame([validated]), metadata["setting"]
    )
    data = engineer(data, metadata["setting"])
    feature_columns = metadata["feature_columns"]
    missing_features = [column for column in feature_columns if column not in data]
    if missing_features:
        raise ValueError(
            "Không thể tạo đặc trưng mô hình yêu cầu: " + ", ".join(missing_features)
        )

    model = artifact["model"]
    features = data[feature_columns]
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Mô hình đã lưu không có lớp Dropout (1).")

    probabilities = model.predict_proba(features)[0]
    dropout_probability = float(probabilities[classes.index(1)])
    prediction = int(model.predict(features)[0])
    return dropout_probability, prediction


def main():
    parser = argparse.ArgumentParser(
        description="Ước lượng nguy cơ bỏ học từ thông tin đầu vào và học kỳ 1."
    )
    parser.add_argument("--model", default="models/best_early_model.joblib")
    args = parser.parse_args()

    artifact = load_artifact(args.model)
    metadata = artifact["metadata"]
    print(
        "Nhập thông tin theo các cột dưới đây. "
        "Mã biến danh mục cần nhập dưới dạng số nguyên; để trống nếu thiếu."
    )
    record = collect_input(input_columns=metadata["input_columns"])
    probability, prediction = predict_record(record, artifact)

    label = "Có nguy cơ bỏ học" if prediction == 1 else "Không bị phân loại là bỏ học"
    print(f"\nĐiểm nguy cơ ước lượng: {probability:.1%}")
    print(f"Nhãn dự đoán: {label}")
    print(
        "Lưu ý: đây là điểm mô hình ước lượng trên dữ liệu UCI, "
        "chưa được hiệu chuẩn và chỉ dùng cho mục đích minh họa."
    )


if __name__ == "__main__":
    main()
