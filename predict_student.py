
import argparse
import math

import joblib
import pandas as pd

from dropout_pipeline import BIN_COLS, CAT_COLS, engineer, normalize_missing_grades


# Tên tiếng Việt hiển thị cho người sử dụng.
VIETNAMESE_COLUMNS = {
    "Marital status": "Tình trạng hôn nhân",
    "Application mode": "Phương thức đăng ký nhập học",
    "Application order": "Thứ tự nguyện vọng đăng ký",
    "Course": "Ngành học",
    "Daytime/evening attendance": "Hình thức học (ban ngày/buổi tối)",
    "Previous qualification": "Trình độ học vấn trước đó",
    "Previous qualification (grade)": "Điểm trung bình của bằng cấp trước đó",
    "Nacionality": "Quốc tịch",
    "Mother's qualification": "Trình độ học vấn của mẹ",
    "Father's qualification": "Trình độ học vấn của cha",
    "Mother's occupation": "Nghề nghiệp của mẹ",
    "Father's occupation": "Nghề nghiệp của cha",
    "Admission grade": "Điểm xét tuyển đầu vào",
    "Displaced": "Sinh viên thuộc diện di dời (0: Không, 1: Có)",
    "Educational special needs": "Nhu cầu giáo dục đặc biệt (0: Không, 1: Có)",
    "Debtor": "Sinh viên đang nợ học phí (0: Không, 1: Có)",
    "Tuition fees up to date": "Đã đóng học phí đúng hạn (0: Không, 1: Có)",
    "Gender": "Giới tính (0: Nữ, 1: Nam)",
    "Scholarship holder": "Sinh viên có học bổng (0: Không, 1: Có)",
    "Age at enrollment": "Tuổi khi nhập học",
    "International": "Sinh viên quốc tế (0: Không, 1: Có)",
    "Curricular units 1st sem (credited)": "Số học phần được công nhận ở học kỳ 1",
    "Curricular units 1st sem (enrolled)": "Số học phần đăng ký ở học kỳ 1",
    "Curricular units 1st sem (evaluations)": "Số lượt đánh giá ở học kỳ 1",
    "Curricular units 1st sem (approved)": "Số học phần đạt ở học kỳ 1",
    "Curricular units 1st sem (grade)": "Điểm trung bình học kỳ 1",
    "Curricular units 1st sem (without evaluations)": "Số học phần không có đánh giá ở học kỳ 1",
    "Curricular units 2nd sem (credited)": "Số học phần được công nhận ở học kỳ 2",
    "Curricular units 2nd sem (enrolled)": "Số học phần đăng ký ở học kỳ 2",
    "Curricular units 2nd sem (evaluations)": "Số lượt đánh giá ở học kỳ 2",
    "Curricular units 2nd sem (approved)": "Số học phần đạt ở học kỳ 2",
    "Curricular units 2nd sem (grade)": "Điểm trung bình học kỳ 2",
    "Curricular units 2nd sem (without evaluations)": "Số học phần không có đánh giá ở học kỳ 2",
    "Unemployment rate": "Tỷ lệ thất nghiệp",
    "Inflation rate": "Tỷ lệ lạm phát",
    "GDP": "Tổng sản phẩm quốc nội (GDP)",
}


def display_name(column):
    """Trả về tên tiếng Việt, giữ tên gốc nếu chưa có bản dịch."""
    return VIETNAMESE_COLUMNS.get(column, column)

CATEGORY_OPTIONS = {
    "Marital status": {
        1: "Độc thân",
        2: "Đã kết hôn",
        3: "Góa",
        4: "Ly hôn",
        5: "Sống chung không kết hôn",
        6: "Ly thân hợp pháp"
    },

    "Application mode": {
        1: "Đợt 1 - Tuyển sinh chung",
        2: "Theo Quy định số 612/93",
        5: "Đợt 1 - Diện đặc biệt (Azores)",
        7: "Đã có bằng đại học khác",
        10: "Theo Quy định số 854-B/99",
        15: "Sinh viên quốc tế",
        16: "Đợt 1 - Diện đặc biệt (Madeira)",
        17: "Đợt 2 - Tuyển sinh chung",
        18: "Đợt 3 - Tuyển sinh chung",
        26: "Theo Quy định 533-A/99, mục b2",
        27: "Theo Quy định 533-A/99, mục b3",
        39: "Trên 23 tuổi",
        42: "Chuyển trường",
        43: "Chuyển ngành",
        44: "Có bằng chuyên môn công nghệ",
        51: "Thay đổi trường hoặc ngành",
        53: "Có bằng đào tạo ngắn hạn",
        57: "Chuyển trường/ngành quốc tế"
    },

    "Course": {
        33: "Công nghệ sản xuất nhiên liệu sinh học",
        171: "Thiết kế đa phương tiện và hoạt hình",
        8014: "Công tác xã hội (buổi tối)",
        9003: "Nông học",
        9070: "Thiết kế truyền thông",
        9085: "Điều dưỡng thú y",
        9119: "Kỹ thuật Tin học",
        9130: "Chăn nuôi ngựa",
        9147: "Quản lý",
        9238: "Công tác xã hội",
        9254: "Du lịch",
        9500: "Điều dưỡng",
        9556: "Vệ sinh răng miệng",
        9670: "Quản lý quảng cáo và tiếp thị",
        9773: "Báo chí và truyền thông",
        9853: "Giáo dục cơ bản",
        9991: "Quản lý (buổi tối)"
    },

    "Previous qualification": {
        1: "Trung học phổ thông",
        2: "Giáo dục đại học - Cử nhân",
        3: "Giáo dục đại học - Bằng đại học",
        4: "Thạc sĩ",
        5: "Tiến sĩ",
        6: "Đang theo học giáo dục đại học",
        9: "Chưa hoàn thành lớp 12",
        10: "Chưa hoàn thành lớp 11",
        12: "Trình độ khác - Lớp 11",
        14: "Lớp 10",
        15: "Chưa hoàn thành lớp 10",
        19: "Giáo dục cơ bản chu kỳ 3",
        38: "Giáo dục cơ bản chu kỳ 2",
        39: "Khóa chuyên môn công nghệ",
        40: "Giáo dục đại học - Bằng chu kỳ 1",
        42: "Khóa kỹ thuật chuyên môn bậc cao",
        43: "Thạc sĩ chu kỳ 2"
    },

    "Nationality": {
        1: "Bồ Đào Nha",
        2: "Đức",
        6: "Tây Ban Nha",
        11: "Ý",
        13: "Hà Lan",
        14: "Anh",
        17: "Litva",
        21: "Angola",
        22: "Cape Verde",
        24: "Guinea",
        25: "Mozambique",
        26: "São Tomé",
        32: "Thổ Nhĩ Kỳ",
        41: "Brazil",
        62: "Romania",
        100: "Moldova",
        101: "Mexico",
        103: "Ukraine",
        105: "Nga",
        108: "Cuba",
        109: "Colombia"
    },

    "Daytime/evening attendance": {
        1: "Học ban ngày",
        0: "Học buổi tối"
    },

    "Gender": {
        1: "Nam",
        0: "Nữ"
    },

    "Displaced": {
        1: "Có",
        0: "Không"
    },

    "Educational special needs": {
        1: "Có",
        0: "Không"
    },

    "Debtor": {
        1: "Có nợ",
        0: "Không nợ"
    },

    "Tuition fees up to date": {
        1: "Đã đóng học phí đúng hạn",
        0: "Chưa đóng học phí đúng hạn"
    },

    "Scholarship holder": {
        1: "Có học bổng",
        0: "Không có học bổng"
    },

    "International": {
        1: "Sinh viên quốc tế",
        0: "Sinh viên trong nước"
    }
}


def show_category_options(column):
    """Hiển thị ý nghĩa các mã phân loại."""
    options = CATEGORY_OPTIONS.get(column)

    if options:
        print("\n  Các lựa chọn:")
        for code, description in options.items():
            print(f"  {code} - {description}")


def parse_input_value(column, value):
    """Kiểm tra và chuyển đổi dữ liệu đầu vào."""
    if value is None or str(value).strip().casefold() in {"", "na", "nan"}:
        return float("nan")

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{display_name(column)}: vui lòng nhập một giá trị số."
        ) from exc

    if not math.isfinite(number):
        raise ValueError(
            f"{display_name(column)}: giá trị phải là số hữu hạn."
        )

    if column in CAT_COLS:
        if not number.is_integer():
            raise ValueError(
                f"{display_name(column)}: mã danh mục phải là số nguyên."
            )
        return int(number)

    if column in BIN_COLS:
        if number not in (0, 1):
            raise ValueError(
                f"{display_name(column)}: chỉ chấp nhận 0 hoặc 1."
            )
        return int(number)

    return number


def collect_input(input_func=input, output_func=print, input_columns=None):
    """Thu thập thông tin đầu vào của sinh viên."""
    if input_columns is None:
        raise ValueError("Thiếu danh sách cột đầu vào trong metadata mô hình.")

    record = {}

    print("\n========== NHẬP THÔNG TIN SINH VIÊN ==========")
    print("Vui lòng nhập dữ liệu theo hướng dẫn bên dưới.")
    print("Nhập 0 hoặc 1 đối với các trường dữ liệu nhị phân.")
    print("Nhấn Enter nếu không có thông tin.\n")

    for index, column in enumerate(input_columns, start=1):
        question = f"{index}. {display_name(column)}"

        while True:
            show_category_options(column)
            value = input_func(f"\n{question} - Nhập mã: ")

            try:
                record[column] = parse_input_value(column, value)
                break
            except ValueError as exc:
                output_func(str(exc))

    return record


def load_artifact(path):
    """Tải mô hình đã huấn luyện."""
    artifact = joblib.load(path)

    if not isinstance(artifact, dict) or not {"model", "metadata"} <= artifact.keys():
        raise ValueError("File mô hình không đúng định dạng.")

    metadata = artifact["metadata"]
    required_metadata = {"setting", "input_columns", "feature_columns"}

    if not isinstance(metadata, dict) or not required_metadata <= metadata.keys():
        raise ValueError("Mô hình thiếu thông tin cần thiết để dự đoán.")

    if metadata["setting"] != "early":
        raise ValueError("Chương trình chỉ hỗ trợ mô hình dự đoán sớm (Early).")

    return artifact


def predict_record(record, artifact):
    """Tiền xử lý và dự đoán nguy cơ bỏ học."""
    metadata = artifact["metadata"]
    input_columns = metadata["input_columns"]

    missing = [column for column in input_columns if column not in record]

    if missing:
        raise ValueError(
            "Thiếu thông tin đầu vào: "
            + ", ".join(display_name(column) for column in missing)
        )

    validated = {
        column: parse_input_value(column, record[column])
        for column in input_columns
    }

    data = normalize_missing_grades(
        pd.DataFrame([validated]), metadata["setting"]
    )

    data = engineer(data, metadata["setting"])

    feature_columns = metadata["feature_columns"]

    missing_features = [
        column for column in feature_columns if column not in data
    ]

    if missing_features:
        raise ValueError(
            "Không thể tạo các đặc trưng cần thiết: "
            + ", ".join(missing_features)
        )

    model = artifact["model"]
    features = data[feature_columns]

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("Mô hình không chứa lớp dự đoán bỏ học (1).")

    probabilities = model.predict_proba(features)[0]
    dropout_probability = float(probabilities[classes.index(1)])

    prediction = int(model.predict(features)[0])

    return dropout_probability, prediction


def main():
    parser = argparse.ArgumentParser(
        description="Dự đoán nguy cơ bỏ học của sinh viên."
    )

    parser.add_argument(
        "--model",
        default="models/best_early_model.joblib",
        help="Đường dẫn đến file mô hình đã huấn luyện."
    )

    args = parser.parse_args()

    print("\n==============================================")
    print("       HỆ THỐNG DỰ ĐOÁN NGUY CƠ BỎ HỌC")
    print("==============================================")

    artifact = load_artifact(args.model)
    metadata = artifact["metadata"]

    record = collect_input(
        input_columns=metadata["input_columns"]
    )

    probability, prediction = predict_record(record, artifact)

    label = (
        "CÓ NGUY CƠ BỎ HỌC"
        if prediction == 1
        else "KHÔNG BỊ PHÂN LOẠI LÀ BỎ HỌC"
    )

    print("\n============== KẾT QUẢ DỰ ĐOÁN ==============")
    print(f"Điểm nguy cơ bỏ học: {probability:.1%}")
    print(f"Kết luận: {label}")
    print("==============================================")

    print(
        "\nLưu ý: Kết quả là xác suất ước lượng từ mô hình "
        "trên dữ liệu UCI, chưa được hiệu chuẩn và chỉ "
        "phục vụ mục đích tham khảo, minh họa."
    )


if __name__ == "__main__":
    main()
