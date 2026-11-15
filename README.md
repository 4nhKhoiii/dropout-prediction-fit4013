# Dự đoán sinh viên có nguy cơ bỏ học

Bài tập lớn học phần **FIT4013 – Nhập môn Học máy** (Năm học 2026–2027), **Đề tài 8**.
Trường Đại học Đại Nam, Khoa Công nghệ Thông tin. Giảng viên hướng dẫn: ThS. Lê Thị Thùy Trang.

## Nhóm thực hiện

| Họ và tên | Mã sinh viên |
|-----------|--------------|
| Ngô Nguyễn Anh Khôi | 1871020336 |
| Hoàng Quang Huy | 1871020300 |
| Vũ Anh Quân | 1871020476 |

## Bài toán

Phân loại nhị phân: từ thông tin lúc nhập học và kết quả học tập, dự đoán sinh viên có **bỏ học** hay không.

- Nhãn `y = 1` nếu `Target = Dropout`; `y = 0` nếu `Target` là `Enrolled` hoặc `Graduate`.
- Hai phạm vi đặc trưng:
  - `early` (phạm vi chính, dùng cho cảnh báo sớm): thông tin nhập học và học kỳ 1.
  - `full` (để so sánh): thêm học kỳ 2. Dữ liệu này đến muộn nên không dùng để cảnh báo sớm.
- Hai luồng tiền xử lý:
  - Luồng A: dữ liệu thô, giữ 3 biến vĩ mô (thất nghiệp, lạm phát, GDP).
  - Luồng B: bỏ 3 biến vĩ mô, thêm đặc trưng tỷ lệ (tỷ lệ môn đạt, tỷ lệ môn không có đánh giá, chênh lệch điểm hai học kỳ ở phạm vi `full`).
- Mô hình: Baseline (Majority), Cây quyết định mặc định, K-NN, Naive Bayes, Cây quyết định tinh chỉnh. Siêu tham số chọn bằng `GridSearchCV` với 5 lớp kiểm định chéo phân tầng, độ đo F1 của lớp bỏ học.

## Dữ liệu

- Nguồn: UCI Machine Learning Repository, dataset 697 – *Predict Students' Dropout and Academic Success* (Realinho et al., 2021).
  https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success
- Quy mô: 4424 mẫu, 37 cột (36 đặc trưng và biến mục tiêu `Target`). Phân bố: Graduate 2209, Dropout 1421 (32,1%), Enrolled 794.
- File dữ liệu nằm ở `data/student_dropout.csv`. Nếu chưa có, tải lại bằng:

```bash
python scripts/download_data.py
```

Script dùng thư viện `ucimlrepo` để tải dataset 697 và lưu vào `data/student_dropout.csv`. File mô tả từng cột (`data_dictionary.csv`) đi kèm bộ dữ liệu trên trang UCI.

## Cài đặt

Yêu cầu Python 3.12 trở lên.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Cách chạy lại kết quả

```bash
# 1. (Nếu cần) tải dữ liệu
python scripts/download_data.py

# 2. Thí nghiệm chính: EDA, tiền xử lý, huấn luyện, đánh giá, phân tích lỗi
python dropout_pipeline.py --data data/student_dropout.csv --out results

# 3. Các hình EDA bổ sung dùng trong báo cáo
python eda_extra.py --data data/student_dropout.csv --out results
```

Kết quả ghi vào `results/`:

| File | Nội dung |
|------|----------|
| `metrics_all.csv` | Chỉ số của mọi mô hình, phạm vi và luồng (CV-F1, Accuracy, Precision, Recall, F1, ROC-AUC, siêu tham số tốt nhất) |
| `error_profile.csv` | Đặc điểm trung bình của nhóm TP/TN/FN/FP (mô hình tốt nhất trong phạm vi `early`, chọn theo CV-F1) |
| `misclassified.csv` | Các mẫu bị dự đoán sai (FN và FP) của mô hình trên |
| `cm_*.png` | Ma trận nhầm lẫn của từng mô hình |
| `importance_*.png` | Độ quan trọng đặc trưng của cây quyết định |
| `eda_*.png` | Các hình khám phá dữ liệu |

Mọi bước học tham số (điền thiếu, chuẩn hóa, mã hóa one-hot) nằm trong `Pipeline` và chỉ fit trên tập huấn luyện. `random_state = 42` được cố định cho việc chia dữ liệu, kiểm định chéo và các mô hình có tính ngẫu nhiên. Chia dữ liệu 80/20 phân tầng theo nhãn (3539 mẫu huấn luyện, 885 mẫu kiểm thử). Tập kiểm thử chỉ dùng một lần để báo cáo kết quả.

## Kết quả chính

Trên tập kiểm thử (885 mẫu), lớp dương là Dropout.

**Phạm vi `early` (cảnh báo sớm)**

| Luồng | Mô hình | F1 | ROC-AUC | Recall |
|-------|---------|----|---------|--------|
| A | Cây quyết định (tinh chỉnh) | 0,748 | 0,881 | 0,810 |
| B | Cây quyết định (tinh chỉnh) | **0,767** | **0,893** | 0,810 |
| B | K-NN | 0,681 | 0,859 | 0,560 |
| B | Naive Bayes | 0,657 | 0,823 | 0,866 |
| – | Majority (baseline) | 0,000 | 0,500 | 0,000 |

**So sánh với phạm vi `full` (F1 / ROC-AUC, Luồng B)**

| Mô hình | early | full |
|---------|-------|------|
| Cây quyết định | 0,767 / 0,893 | 0,768 / 0,917 |
| K-NN | 0,681 / 0,859 | 0,723 / 0,889 |
| Naive Bayes | 0,657 / 0,823 | 0,695 / 0,847 |

Nhận xét ngắn:
- Cây quyết định tinh chỉnh cho F1 và ROC-AUC cao nhất ở phạm vi `early`.
- Luồng B cao hơn Luồng A ở cả ba thuật toán, nhưng mức chênh nhỏ (0,007–0,021 F1) và chỉ dựa trên một lần chia dữ liệu.
- Thêm học kỳ 2 tăng ROC-AUC 0,018–0,030 nhưng đến quá muộn để can thiệp.

Chi tiết và các phân tích khác nằm trong báo cáo (`report/`).

## Cấu trúc thư mục

```
.
├── data/
│   └── student_dropout.csv       # Dữ liệu UCI 697 (có thể tải lại bằng scripts/download_data.py)
├── scripts/
│   └── download_data.py          # Tải dữ liệu bằng ucimlrepo
├── dropout_pipeline.py           # Pipeline chính: EDA, huấn luyện, đánh giá, phân tích lỗi
├── eda_extra.py                  # Sinh các hình EDA bổ sung cho báo cáo
├── requirements.txt              # Phiên bản thư viện để tái lập kết quả
├── results/                      # Bảng và hình kết quả
├── report/
│   └── bao_cao_de_tai_8.tex      # Báo cáo LaTeX (biên dịch bằng XeLaTeX)
├── AI_USAGE.md                   # Khai báo sử dụng công cụ AI
└── README.md
```

## Hạn chế

- Dữ liệu từ một cơ sở đào tạo ở Bồ Đào Nha, nên khó tổng quát hóa sang môi trường khác.
- Mỗi cấu hình chỉ chạy với một `random_state`, nên chưa đánh giá được độ ổn định của kết quả.
- Ngưỡng quyết định giữ mặc định 0,5, chưa điều chỉnh theo chi phí của bỏ sót và cảnh báo nhầm.
- Chưa lưu mô hình ra file và chưa có giao diện dự đoán cho hồ sơ mới.

## Thành viên và phân công

| Thành viên | Nhiệm vụ |
|------------|----------|
| Ngô Nguyễn Anh Khôi | *(điền)* |
| Hoàng Quang Huy | *(điền)* |
| Vũ Anh Quân | *(điền)* |

## Khai báo sử dụng AI

Xem [AI_USAGE.md](AI_USAGE.md).

## Tài liệu tham khảo

- Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2021). *Predict Students' Dropout and Academic Success*. UCI Machine Learning Repository.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
