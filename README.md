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

## Tính tái lập và thiết kế thí nghiệm

### Hằng số toàn cục

| Hằng số | Giá trị | Ý nghĩa |
|---------|---------|----------|
| `RANDOM_STATE` | `42` | `random_state` cho `train_test_split`, `StratifiedKFold`, `DecisionTreeClassifier`, và `np.random.seed` |
| `TEST_SIZE` | `0.2` | Tỷ lệ tập kiểm thử (20 %) |
| `CV_FOLDS` | `5` | Số fold trong kiểm định chéo phân tầng (`StratifiedKFold`) |

### Phân chia dữ liệu

```
Tổng mẫu (sau làm sạch): 4 424
  → Huấn luyện : 3 539 mẫu (80 %, dùng để fit pipeline + GridSearchCV)
  → Kiểm thử   :   885 mẫu (20 %, chỉ dùng một lần để báo cáo chỉ số cuối)
```

Phân chia phân tầng theo nhãn (`stratify=y`) nên tỷ lệ Dropout ≈ 32 % được giữ nguyên ở cả hai tập.

### Lưới siêu tham số tìm kiếm (GridSearchCV)

| Mô hình | Tham số tìm kiếm | Miền giá trị |
|---------|------------------|--------------|
| **Cây quyết định (tinh chỉnh)** | `max_depth` | `[3, 5, 7, 10, None]` |
| | `min_samples_leaf` | `[5, 10, 20, 50]` |
| | `criterion` | `["gini", "entropy"]` |
| | `class_weight` | `[None, "balanced"]` |
| **K-NN** | `n_neighbors` | `[3, 5, 9, 15, 21]` |
| | `weights` | `["uniform", "distance"]` |
| **Naive Bayes** | `var_smoothing` | `[1e-9, 1e-7, 1e-5, 1e-3]` |
| **Baseline Majority** | *(không có)* | – |
| **Cây quyết định (mặc định)** | *(không có)* | – |

Siêu tham số tốt nhất của từng thí nghiệm được lưu vào cột `best_params` trong `results/metrics_all.csv`.

### Bảo đảm không rò rỉ dữ liệu (no data leakage)

| Bước tiền xử lý | Phương án thực hiện | Trạng thái |
|-----------------|---------------------|------------|
| Điền giá trị thiếu (`SimpleImputer`) | Nằm trong `sklearn.Pipeline`, chỉ `.fit()` trên tập huấn luyện | ✅ |
| Chuẩn hóa (`StandardScaler`) | Nằm trong `sklearn.Pipeline`, chỉ `.fit()` trên tập huấn luyện | ✅ |
| Mã hóa phân loại (`OneHotEncoder`) | Nằm trong `sklearn.Pipeline`, chỉ `.fit()` trên tập huấn luyện | ✅ |
| Kỹ thuật đặc trưng (`engineer()`) | Chỉ dùng phép tính từng dòng (tỷ lệ, hiệu số), không học tham số từ dữ liệu | ✅ |
| Chọn siêu tham số (`GridSearchCV`) | CV nội bộ hoàn toàn trên tập huấn luyện; tập kiểm thử không tham gia | ✅ |
| Chọn mô hình để phân tích lỗi | Chọn theo CV-F1, không dùng chỉ số tập kiểm thử | ✅ |

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

# 3. Demo dự đoán (sau khi đã huấn luyện và lưu mô hình)
python predict_student.py --model models/best_early_model.joblib

# 4. Các hình EDA bổ sung dùng trong báo cáo
python eda_extra.py --data data/student_dropout.csv --out results

# 5. Rà soát tương quan và nguy cơ rò rỉ dữ liệu (chạy riêng)
python leakage_check.py --data data/student_dropout.csv --out results
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
| `leakage_report.txt`, `leakage_*.png` | Báo cáo và biểu đồ rà soát tương quan/nguy cơ leakage (tạo bởi `leakage_check.py`) |

Mô hình `early` có CV-F1 cao nhất (không tính baseline) được chọn mà không dùng chỉ số test và lưu thành `models/best_early_model.joblib`. File chứa cả bước tiền xử lý và classifier; thư mục `models/` được bỏ qua bởi Git. Chạy `dropout_pipeline.py` để tạo/cập nhật artifact trước khi chạy demo.

### Demo nhập hồ sơ sinh viên

`predict_student.py` hỏi lần lượt các trường đầu vào cần thiết của mô hình đã lưu: thông tin nhập học và kết quả học kỳ 1. Mã các biến danh mục cần nhập dưới dạng số nguyên theo quy ước mã hóa của bộ dữ liệu UCI; các biến nhị phân chỉ nhận `0` hoặc `1`. Có thể để trống trường chưa biết; nếu không có đánh giá học kỳ 1, điểm được xem là thiếu như trong dữ liệu huấn luyện và sẽ được điền bởi pipeline. Chương trình áp dụng lại feature engineering của luồng mô hình và in nhãn dự đoán cùng điểm nguy cơ ước lượng. Mô hình hiện được chọn theo CV-F1 trong phạm vi `early`; việc chọn mô hình không dựa trên tập test.

Điểm `predict_proba` chưa được hiệu chuẩn thành tỷ lệ bỏ học ngoài thực tế. Dữ liệu UCI đến từ một cơ sở đào tạo ở Bồ Đào Nha, do đó kết quả chỉ phù hợp cho mục đích minh họa, không dùng làm căn cứ ra quyết định về sinh viên.

Mọi bước học tham số (điền thiếu, chuẩn hóa, mã hóa one-hot) nằm trong `Pipeline` và chỉ fit trên tập huấn luyện. `random_state = 42` được cố định cho việc chia dữ liệu, kiểm định chéo và các mô hình có tính ngẫu nhiên. Chia dữ liệu 80/20 phân tầng theo nhãn (3 539 mẫu huấn luyện, 885 mẫu kiểm thử). Tập kiểm thử chỉ dùng một lần để báo cáo kết quả. Xem chi tiết lưới siêu tham số và bảo đảm không rò rỉ dữ liệu tại mục **[Tính tái lập và thiết kế thí nghiệm](#tính-tái-lập-và-thiết-kế-thí-nghiệm)** ở trên.

## Đối chiếu checklist 7 giai đoạn

Đối chiếu với mã nguồn và kết quả đã lưu. Trạng thái “một phần” nghĩa là có đầu ra liên quan nhưng chưa đáp ứng đầy đủ phạm vi/yêu cầu của checklist.

| Checklist | Trạng thái | Thực tế hiện tại / phần còn thiếu |
|-----------|------------|----------------------------------|
| 0.1 Load raw data | ✅ | Đọc CSV, chuẩn hóa tên cột, kiểm tra các cột bắt buộc và nhãn hợp lệ; bỏ dòng trùng lặp. |
| 0.2 Drop ID/leakage, ghi lý do | 🟡 Một phần | Dataset không có ID riêng; `Target` không nằm trong `X` vì feature columns được liệt kê tường minh. `early` không dùng đặc trưng học kỳ 2; `full` cố ý giữ để so sánh hồi cứu. Báo cáo `leakage_check.py` có phân tích lý do theo nhóm biến, nhưng không thực hiện một bước drop cột có log lý do trong pipeline. Báo cáo này cũng phân tích toàn bộ dữ liệu. |
| 0.3 Train/test split | ✅ | `test_size=0.2`, `stratify=y`, `random_state=42`; cùng cách chia được áp dụng cho từng phạm vi/luồng. |
| 1.1 Shape, dtypes, missing, duplicated, nhãn | 🟡 Một phần | Có shape, số ô thiếu tổng cộng, số dòng trùng bị loại và phân phối nhãn. Chưa in `dtypes` hoặc missing theo từng cột. |
| 1.2 `describe()` | ❌ | Chưa có bảng thống kê mô tả biến số. |
| 1.3 Histogram, boxplot | 🟡 Một phần | Có histogram và boxplot một số biến trong EDA chính, cùng các biểu đồ bổ sung; chưa khảo sát có hệ thống các biến số/outlier. Quan trọng: hiện `eda(df, ...)` và `eda_extra.py` chạy trên toàn bộ dữ liệu trước split, không phải chỉ `X_train`. |
| 1.4 Tần suất biến phân loại | ❌ | Chưa có bar chart tần suất các biến phân loại. Biểu đồ tỷ lệ Dropout theo một số biến nhị phân không thay thế biểu đồ tần suất. |
| 1.5 Correlation/leakage | 🟡 Một phần | `leakage_check.py` tạo heatmap/tương quan và giải thích rủi ro học kỳ 2, nhưng chạy trên toàn bộ dataset trước split; cần giới hạn EDA/khám phá này ở train nếu tuân thủ checklist nghiêm ngặt. |
| 1.6 Insight cho Pipeline B | 🟡 Một phần | README và leakage report nêu lý do thử đặc trưng tỷ lệ, bỏ biến vĩ mô; chưa có ghi nhận insight theo một quy trình EDA chỉ trên train, và B thay đổi đồng thời nhiều yếu tố nên chưa tách được tác động. |
| 2.1–2.2 Numeric/categorical transformer | ✅ / 🟡 | Có median imputer cho số; mode imputer và One-Hot với `handle_unknown="ignore"` cho categorical. `StandardScaler` chỉ bật cho KNN/Naive Bayes (hợp lý vì cây không cần scale), không bật cho mọi numeric transformer như checklist mẫu. |
| 2.3 Pipeline A chỉ có `pre` | 🟡 | Tiền xử lý được đóng trong `Pipeline` cùng estimator ở `fit_one`; chưa khai báo `A_pipe` độc lập chỉ gồm bước `pre`. Cách hiện tại vẫn đảm bảo preprocessing được fit bên trong CV/train, nhưng cấu trúc không trùng checklist từng bước. |
| 3.1 Pipeline B cải tiến | ✅ / 🟡 | B thêm tỷ lệ môn đạt và tỷ lệ môn không có đánh giá; ở `full` còn thêm chênh lệch điểm HK2–HK1; đồng thời bỏ 3 biến vĩ mô. Không dùng `KNNImputer` hay `SelectKBest`; đó không phải thiếu nếu feature engineering là hướng cải tiến đã chọn. |
| 3.2 Giả thuyết trong comment | 🟡 | Có giải thích trong README và một phần trong comment về bỏ biến vĩ mô, nhưng chưa ghi giả thuyết rõ cho từng feature engineering ngay cạnh phần tạo feature trong code. |
| 3.3 Pipeline B + selector | 🟡 | Pipeline mô hình B được tạo cùng cấu trúc `prep` + `clf` như A trên bộ cột/feature khác; không có bước `select` vì hướng B đang chọn là feature engineering, không phải feature selection. |
| 4.1–4.2 Ma trận A/B × KNN/NB/DT | ✅ | Đã chạy đủ sáu tổ hợp trong cả `early` và `full`; còn chạy thêm baseline Majority và cây mặc định. |
| 4.3 GridSearchCV/CV/F1 và lưu kết quả | ✅ / 🟡 | Dùng `GridSearchCV`, `StratifiedKFold(5)` và `scoring="f1"` cho các mô hình cần tune. `best_params` và CV-F1 được lưu trong `metrics_all.csv` (`best_params`/`cv_f1`), tương ứng với `best_params_`/`best_score_`. KNN tune `n_neighbors`, `weights` nhưng chưa tune `metric`; NB tune `var_smoothing`; DT tune `criterion`, `max_depth`, `min_samples_leaf`, `class_weight`. |
| 4.4 KMeans, elbow, silhouette (phụ) | ➖ | Không triển khai. Đây là mục phụ/khám phá cụm, không ảnh hưởng việc hoàn thành ma trận phân loại 2×3. |
| 5.1 Metrics và confusion matrix | ✅ | Mỗi cấu hình có Accuracy, Precision, Recall, F1 và ROC-AUC trong `metrics_all.csv`, cùng confusion matrix. Thứ tự ma trận theo sklearn là `[[TN, FP], [FN, TP]]`. |
| 5.2 Bảng so sánh chéo | 🟡 Một phần | CSV có các chỉ số từng dòng; console có pivot F1 theo mô hình/phạm vi/luồng. Chưa tạo bảng pivot đầy đủ tất cả metrics với cột A/B × metric. |
| 5.3 Heatmap confusion matrix | ✅ | Đã xuất confusion matrix dạng ô màu bằng `ConfusionMatrixDisplay` cho tất cả mô hình, rộng hơn yêu cầu chỉ vẽ các ô nổi bật. |
| 6.1–6.2 Trích và mô tả FN/FP | 🟡 | `misclassified.csv` và `error_profile.csv` lưu mẫu và trung bình một số đặc trưng cho FN/FP, nhưng chỉ cho một mô hình `early` được chọn theo CV-F1; chưa phân tích thống kê/đặc trưng riêng cho mọi ô. |
| 6.3 So sánh số FN A/B | 🟡 | README đã đối chiếu số FN của KNN và Decision Tree trên cùng test set; pipeline chưa xuất bảng so sánh FN/FP tự động cho sáu tổ hợp. |
| 6.4 Kết luận theo bối cảnh giáo dục | ✅ / 🟡 | README nêu FN có thể khiến bỏ lỡ sinh viên cần hỗ trợ và cần ưu tiên giảm FN; pipeline chưa tự sinh kết luận này và B chưa chứng minh đã giảm FN. |
| 7.1 Random state/test size/CV/grid | ✅ | Các hằng số, cách chia và grid được ghi trong README; kết quả từng cấu hình được lưu trong CSV. |
| 7.2 Chạy lại cùng kết quả | 🟡 | Có seed cố định và phiên bản thư viện ghim trong `requirements.txt`; chưa xác nhận bằng một lần chạy lại độc lập và so sánh output. Repository chạy bằng script, không có notebook. |

### Cảnh báo về thứ tự EDA và chia dữ liệu

Checklist yêu cầu tách dữ liệu trước rồi chỉ làm EDA trên train. Hiện tại `main()` gọi EDA trên toàn bộ dataframe trước `run_experiments()`, và `leakage_check.py`/`eda_extra.py` cũng đọc toàn bộ file. Các bước này không fit scaler/imputer/model nên không làm rò rỉ tham số huấn luyện trực tiếp; tuy nhiên, nếu dùng thông tin từ test để chọn feature hay hướng cải tiến thì có thể tạo leakage trong quá trình ra quyết định. Đây là thiếu sót quan trọng nhất so với checklist mới.

### So sánh FN hiện có (`early`)

Số FN dưới đây được đối chiếu từ confusion matrix trên cùng test split; B hiện chưa chứng minh giảm FN:

| Luồng | Mô hình | Recall Dropout | F1 | FN |
|-------|---------|----------------:|---:|---:|
| A | KNN | 0,563 | 0,674 | 124 |
| B | KNN | 0,560 | 0,681 | 125 |
| A | Decision Tree | 0,810 | 0,748 | 54 |
| B | Decision Tree | 0,810 | 0,767 | 54 |

B tăng F1 trong hai cặp này nhưng KNN có thêm 1 FN và Decision Tree có số FN không đổi. Trong bối cảnh hỗ trợ sinh viên, FN (bỏ sót người thực sự bỏ học) có thể nghiêm trọng hơn FP (cảnh báo nhầm), nên cần ưu tiên Recall/giảm FN khi chọn ngưỡng và mô hình; đánh giá triển khai cũng cần xem xét chi phí cảnh báo nhầm và xác nhận trên dữ liệu phù hợp.

### Các việc còn thiếu đáng ưu tiên

1. Tách train/test trước EDA; chỉ dùng `X_train`/`y_train` để mô tả, vẽ biểu đồ, rà tương quan và quyết định hướng cải tiến. Giữ test set cho đánh giá cuối.
2. Bổ sung EDA gồm `dtypes`, missing theo cột, `describe()`, biểu đồ tần suất categorical và khảo sát numeric/outlier; ghi lại insight dùng để thiết kế B.
3. Ghi giả thuyết feature engineering của B trong comment gần chỗ tạo feature; nếu muốn biết yếu tố nào giúp, chạy ablation tách feature mới khỏi việc bỏ biến vĩ mô.
4. Thêm `metric` vào lưới KNN nếu phù hợp với mục tiêu; tạo bảng pivot cho đầy đủ Accuracy/Precision/Recall/F1 và TN/FP/FN/TP cho sáu cấu hình.
5. Tự động lưu FN/FP, `classification_report` và hồ sơ lỗi theo từng cấu hình để so sánh A/B; hiện phân tích lỗi chi tiết chỉ dành cho một mô hình tốt nhất.
6. Chạy lại pipeline độc lập và so sánh CSV/kết quả để xác nhận khả năng tái lập. KMeans là phần phụ, chỉ cần làm nếu đề tài yêu cầu phân tích cụm.

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
├── predict_student.py            # Demo terminal dự đoán từ hồ sơ mới
├── models/                       # Pipeline tốt nhất đã lưu (tự sinh, không commit)
├── eda_extra.py                  # Sinh các hình EDA bổ sung cho báo cáo
├── leakage_check.py              # Rà soát tương quan và nguy cơ data leakage
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
- Điểm xác suất chưa được hiệu chuẩn trên dữ liệu ngoài mẫu; demo terminal chỉ dùng để minh họa, chưa có giao diện web.


## Tài liệu tham khảo

- Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2021). *Predict Students' Dropout and Academic Success*. UCI Machine Learning Repository.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
