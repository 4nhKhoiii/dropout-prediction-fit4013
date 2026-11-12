# Dự đoán sinh viên có nguy cơ bỏ học

Bài tập lớn học phần **FIT4013 – Nhập môn Học máy** (Năm học 2026–2027), **Đề tài 8**.
Trường Đại học Đại Nam, Khoa Công nghệ Thông tin. Giảng viên hướng dẫn: ThS. Lê Thị Thùy Trang.

Bài toán phân loại nhị phân: từ thông tin lúc nhập học và sau học kỳ 1, dự đoán sinh viên có bỏ học hay không (`Dropout` = 1, `Enrolled` và `Graduate` = 0).

## Nhóm thực hiện
- Ngô Nguyễn Anh Khôi – 1871020336
- Hoàng Quang Huy – 1871020300
- Vũ Anh Quân – 1871020476

## Dữ liệu
- Nguồn: UCI Machine Learning Repository, dataset 697 — *Predict Students' Dropout and Academic Success* (Realinho et al., 2021).
- Link: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- Quy mô: 4424 mẫu, 37 cột (36 đặc trưng + biến mục tiêu `Target`).
- Cách dùng: tải bộ dữ liệu, đổi tên thành `data.csv` và đặt ở thư mục gốc của repo. File này **không** được commit vào repo do dung lượng. File mô tả từng cột (`data_dictionary.csv`) đi kèm bộ dữ liệu trên UCI.

## Cài đặt
Yêu cầu Python 3.12 trở lên.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Cách chạy lại kết quả
```bash
# 1. Đặt data.csv ở thư mục gốc
# 2. Thí nghiệm chính: EDA, tiền xử lý, huấn luyện, đánh giá, phân tích lỗi
python dropout_pipeline.py --data data.csv --out results

# 3. Các hình EDA bổ sung dùng trong báo cáo (mục 1.4)
python eda_extra.py --data data.csv --out results
```
Kết quả được ghi vào `results/` (bảng `metrics_all.csv`, `error_profile.csv`, `misclassified.csv` và các hình PNG). Mọi bước học tham số đều nằm trong `Pipeline` và chỉ fit trên tập huấn luyện. `random_state = 42` được cố định cho chia dữ liệu và kiểm định chéo.

## Cấu trúc thư mục
```
.
├── dropout_pipeline.py     # Pipeline chính: EDA, huấn luyện, đánh giá, phân tích lỗi
├── eda_extra.py            # Sinh các hình EDA bổ sung cho báo cáo
├── requirements.txt        # Phiên bản thư viện đã dùng để tái lập kết quả
├── results/                # Bảng và hình kết quả
├── report/
│   └── bao_cao_de_tai_8.tex  # Báo cáo LaTeX (biên dịch bằng XeLaTeX)
├── AI_USAGE.md             # Khai báo sử dụng công cụ AI
└── README.md
```

## Kết quả chính (phạm vi `early`, tập kiểm thử 885 mẫu)

| Luồng | Mô hình | F1 | ROC-AUC | Recall |
|-------|---------|----|---------|--------|
| A | Cây quyết định (đã tinh chỉnh) | 0.748 | 0.881 | 0.810 |
| B | Cây quyết định (đã tinh chỉnh) | **0.767** | **0.893** | 0.810 |
| B | K-NN | 0.681 | 0.859 | 0.560 |
| B | Naive Bayes | 0.657 | 0.823 | 0.866 |
| – | Majority (baseline) | 0.000 | 0.500 | 0.000 |

Chi tiết và các phân tích khác nằm trong báo cáo.

## Thành viên và phân công
| Thành viên | Nhiệm vụ |
|------------|----------|
| Ngô Nguyễn Anh Khôi | |
| Hoàng Quang Huy | |
| Vũ Anh Quân | |

## Khai báo AI
Xem [AI_USAGE.md](AI_USAGE.md).
