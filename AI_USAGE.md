# Khai báo sử dụng công cụ AI

Nhóm ghi rõ cách đã sử dụng công cụ AI trong quá trình thực hiện đề tài.

| Thành viên | Công cụ | Mục đích sử dụng | Phần việc liên quan | Ghi chú kiểm chứng |
|------------|---------|------------------|---------------------|--------------------|
| Ngô Nguyễn Anh Khôi | Trợ lý lập trình và mô hình ngôn ngữ | Gợi ý cấu trúc mã nguồn, cách dùng `Pipeline` trong scikit-learn | `dropout_pipeline.py`, `eda_extra.py` | Chạy lại toàn bộ thí nghiệm và so khớp với `results/` |
| Hoàng Quang Huy | | | | |
| Vũ Anh Quân |Trợ lý lập trình AI | Hỗ trợ chuẩn hóa nguồn dữ liệu UCI, tối ưu khối phân tích lỗi (Error Analysis), rà soát nguy cơ data leakage|scripts/download_data.py, dropout_pipeline.py, leakage_check.py, README.md, report/bao_cao_de_tai_8.tex | Đã kiểm tra git diff, chạy lại pipeline sinh kết quả (metrics_all.csv, misclassified.csv) và đối chiếu trực tiếp các chỉ số TP/TN/FP/FN, báo cáo rà soát leakage|

**Nguyên tắc:** nhóm chịu trách nhiệm về tính đúng đắn của nội dung, mã nguồn và số liệu, và phải giải thích được mọi phần đã nộp.
