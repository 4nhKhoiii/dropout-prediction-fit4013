"""Sinh các hình EDA bổ sung cho báo cáo (phần 1.4).

Chạy:  python eda_extra.py --data data.csv --out results
Kết quả: eda_hist_numeric.png, eda_box_by_target.png, eda_rate_binary.png, eda_corr.png
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402


def load(path):
    """Đọc CSV tự nhận diện dấu phân cách (, hoặc ;) và chuẩn hóa cột."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {path}. Hãy kiểm tra tham số --data."
        )

    d = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    d.columns = [
        str(c).strip().replace("\\ufeff", "").replace("Nacionality", "Nationality")
        for c in d.columns
    ]

    target_col = next((c for c in d.columns if c.lower() == "target"), None)
    if target_col is None:
        raise ValueError(
            "Không tìm thấy cột Target. Hãy kiểm tra dấu phân cách và tên cột. "
            f"Các cột đọc được: {d.columns.tolist()}"
        )
    if target_col != "Target":
        d.rename(columns={target_col: "Target"}, inplace=True)

    d["Target"] = d["Target"].astype(str).str.strip()
    d["y"] = (d["Target"].str.lower() == "dropout").astype(int)

    required = [
        "Age at enrollment", "Admission grade",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Gender", "Debtor", "Tuition fees up to date",
        "Scholarship holder", "Displaced",
        "Previous qualification (grade)", "Application mode",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 2nd sem (enrolled)",
        "Unemployment rate", "Inflation rate", "GDP",
    ]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(
            "Dataset thiếu các cột cần cho biểu đồ EDA:\\n- "
            + "\\n- ".join(missing)
            + f"\\nCác cột hiện có: {d.columns.tolist()}"
        )

    for col in required:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    print(f"Đã đọc dữ liệu: {d.shape[0]} dòng, {d.shape[1]} cột")
    print("Phân bố Target:\\n", d["Target"].value_counts().to_string())
    return d


def hist_numeric(d, out):
    cols = ["Age at enrollment", "Admission grade",
            "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
            "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)"]
    labels = {0: "Không bỏ học", 1: "Bỏ học"}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, c in zip(axes.ravel(), cols):
        for k, col in [(0, "tab:blue"), (1, "tab:red")]:
            ax.hist(d.loc[d.y == k, c], bins=25, alpha=0.55, color=col,
                    label=labels[k], density=True)
        ax.set_title(c, fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Phân phối các biến số chính theo nhãn (mật độ chuẩn hóa)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "eda_hist_numeric.png"), dpi=200)
    plt.close(fig)


def box_target(d, out):
    order = ["Graduate", "Enrolled", "Dropout"]
    cols = ["Curricular units 1st sem (approved)",
            "Curricular units 2nd sem (approved)",
            "Curricular units 1st sem (grade)"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, c in zip(axes, cols):
        sns.boxplot(data=d, x="Target", y=c, order=order, ax=ax,
                    color="lightgray", fliersize=2)
        ax.set_title(c.replace("Curricular units ", ""), fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(labelsize=8)
    fig.suptitle("Số môn đạt và điểm trung bình theo nhãn", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "eda_box_by_target.png"), dpi=200)
    plt.close(fig)


def rate_binary(d, out):
    items = [("Gender", {0: "Nữ", 1: "Nam"}),
             ("Debtor", {0: "Không nợ", 1: "Có nợ"}),
             ("Tuition fees up to date", {0: "Chưa đóng đủ", 1: "Đóng đúng hạn"}),
             ("Scholarship holder", {0: "Không học bổng", 1: "Có học bổng"}),
             ("Displaced", {0: "Không", 1: "Có"})]
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.6), sharey=True)
    for ax, (c, m) in zip(axes, items):
        g = d.groupby(c)["y"].agg(["mean", "count"]).reset_index()
        labels = [f"{m[v]}\n(n={n})" for v, n in zip(g[c], g["count"])]
        ax.bar(labels, g["mean"] * 100, color=["tab:blue", "tab:red"])
        ax.set_title(c, fontsize=9)
        ax.tick_params(labelsize=7)
        for i, v in enumerate(g["mean"] * 100):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=8)
    axes[0].set_ylabel("Tỷ lệ bỏ học (%)")
    fig.suptitle("Tỷ lệ bỏ học theo một số biến nhị phân", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "eda_rate_binary.png"), dpi=200)
    plt.close(fig)


def corr_heatmap(d, out):
    num = ["Age at enrollment", "Admission grade", "Previous qualification (grade)",
           "Application mode", "Curricular units 1st sem (enrolled)",
           "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
           "Curricular units 1st sem (grade)", "Curricular units 2nd sem (enrolled)",
           "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)",
           "Unemployment rate", "Inflation rate", "GDP", "y"]
    short = {c: c.replace("Curricular units ", "CU ").replace(" (", "\n(") for c in num}
    num = [c for c in num if c in d.columns]
    for c in num:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    cm = d[num].corr(method="spearman")
    if cm.empty:
        raise ValueError("Không có cột số hợp lệ để tạo biểu đồ tương quan.")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(cm.rename(index=short, columns=short), cmap="RdBu_r", vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 6},
                cbar_kws={"shrink": 0.7}, ax=ax)
    ax.tick_params(labelsize=7)
    ax.set_title("Tương quan Spearman giữa các biến số và nhãn bỏ học (y)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "eda_corr.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/student_dropout.csv")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = load(args.data)
    hist_numeric(d, args.out)
    box_target(d, args.out)
    rate_binary(d, args.out)
    corr_heatmap(d, args.out)
    print("Đã lưu 4 hình EDA vào", args.out)


if __name__ == "__main__":
    main()
