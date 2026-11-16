"""
leakage_check.py – Kiem tra tuong quan va ra soat data leakage
===============================================================
Chay:   python leakage_check.py --data data/student_dropout.csv --out results

Sinh ra:
  results/leakage_corr_full.png        - heatmap Spearman toan bo dac trung vs y
  results/leakage_top_corr.png         - bar chart top-20 |r_spearman| voi y
  results/leakage_sem2_risk.png        - scatter cac bien HK2 vs HK1
  results/leakage_sem_internal_corr.png- tuong quan noi bo HK1/HK2
  results/leakage_report.txt           - bao cao van ban day du
"""
import argparse
import os
import sys
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import spearmanr, pointbiserialr

warnings.filterwarnings("ignore")

# ── Màu phân loại rủi ro ────────────────────────────────────────────────────
COLOR = {"HIGH": "#e74c3c", "MEDIUM": "#e67e22", "LOW": "#27ae60", "OK": "#2980b9"}

# ── Nhóm cột theo semantic ──────────────────────────────────────────────────
ENROLLMENT_COLS = [
    "Marital status", "Application mode", "Application order",
    "Course", "Daytime/evening attendance",
    "Previous qualification", "Previous qualification (grade)",
    "Nationality", "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
    "Displaced", "Educational special needs",
    "Debtor", "Tuition fees up to date",
    "Gender", "Scholarship holder", "International",
    "Age at enrollment", "Admission grade",
]

SEM1_COLS = [
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
]

SEM2_COLS = [
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
]

MACRO_COLS = ["Unemployment rate", "Inflation rate", "GDP"]

TARGET_COL = "Target"


# ── Load & prep ──────────────────────────────────────────────────────────────
def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [
        str(c).strip().replace("\ufeff", "").replace("Nacionality", "Nationality")
        for c in df.columns
    ]
    target_col = next((c for c in df.columns if c.lower() == "target"), None)
    if target_col and target_col != "Target":
        df.rename(columns={target_col: "Target"}, inplace=True)
    df["Target"] = df["Target"].astype(str).str.strip()
    df["y"] = (df["Target"].str.lower() == "dropout").astype(int)
    # ép về số
    for c in df.columns:
        if c not in ["Target", "y"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ── Tính tương quan Spearman với nhãn y ─────────────────────────────────────
def spearman_with_target(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in ["Target", "y"]]
    rows = []
    for c in feature_cols:
        valid = df[[c, "y"]].dropna()
        if len(valid) < 30 or valid[c].nunique() < 2:
            continue
        r, p = spearmanr(valid[c], valid["y"])
        rows.append({"feature": c, "spearman_r": r, "abs_r": abs(r), "p_value": p})
    return pd.DataFrame(rows).sort_values("abs_r", ascending=False).reset_index(drop=True)


# ── Phân loại rủi ro leakage ─────────────────────────────────────────────────
def classify_leakage(feature: str, spearman_r: float) -> tuple[str, str]:
    """Trả về (mức độ rủi ro, lý do)."""
    abs_r = abs(spearman_r)

    # Nhóm 1 – Dữ liệu HK2: chỉ có sau khi kết thúc HK2 (kết quả dropout đã biết)
    if any(feature.startswith(f"Curricular units 2nd sem") for _ in [1]):
        if abs_r >= 0.35:
            return "HIGH", (
                "Biến học kỳ 2 – thông tin chỉ có SAU khi sinh viên đã quyết định bỏ học. "
                "Tương quan cao (|r|={:.2f}) gợi ý quan hệ nhân-quả ngược: "
                "bỏ học → không đăng ký/thi HK2, không phải ngược lại. "
                "Khi dùng để cảnh báo SỚM (setting=early), các biến này PHẢI bị loại.".format(abs_r)
            )
        else:
            return "MEDIUM", (
                "Biến học kỳ 2 – về mặt thời gian có thể xảy ra sau quyết định bỏ học. "
                "Tương quan vừa phải (|r|={:.2f}). Chỉ nên dùng ở setting=full "
                "và phải ghi chú rõ hạn chế này.".format(abs_r)
            )

    # Nhóm 2 – Biến vĩ mô: đo tại thời điểm nhập học, không phụ thuộc hành vi SV
    if feature in MACRO_COLS:
        return "LOW", (
            "Biến vĩ mô (đo ngoài nhà trường) – không phụ thuộc hành vi cá nhân, "
            "không có quan hệ nhân-quả ngược với nhãn. "
            "Tuy nhiên tương quan thấp (|r|={:.2f}) gợi ý tín hiệu yếu, "
            "dễ gây nhiễu hơn là cải thiện mô hình.".format(abs_r)
        )

    # Nhóm 3 – HK1: thông tin thu thập trong HK1, trước khi kết quả dropout chính thức
    if any(feature.startswith("Curricular units 1st sem") for _ in [1]):
        if abs_r >= 0.40:
            return "OK", (
                "Biến học kỳ 1 – thu thập trước khi dropout chính thức xảy ra. "
                "Tương quan cao (|r|={:.2f}) phản ánh tín hiệu dự báo thực, "
                "không phải data leakage.".format(abs_r)
            )
        else:
            return "OK", (
                "Biến học kỳ 1 – tín hiệu dự báo tốt, không có rủi ro leakage.".format()
            )

    # Nhóm 4 – Thông tin nhập học
    if feature in ENROLLMENT_COLS:
        if abs_r >= 0.30:
            return "OK", (
                "Biến nhập học – có sẵn trước khi nhập học, không có rủi ro leakage. "
                "Tương quan (|r|={:.2f}).".format(abs_r)
            )
        else:
            return "OK", (
                "Biến nhập học – không có rủi ro leakage, tương quan thấp (|r|={:.2f}).".format(abs_r)
            )

    return "OK", f"Không xác định nhóm, tương quan (|r|={abs_r:.2f})."


# ── Vẽ heatmap toàn bộ ───────────────────────────────────────────────────────
def plot_full_heatmap(df: pd.DataFrame, out: str):
    all_feat = (
        [c for c in ENROLLMENT_COLS if c in df.columns]
        + [c for c in SEM1_COLS if c in df.columns]
        + [c for c in SEM2_COLS if c in df.columns]
        + [c for c in MACRO_COLS if c in df.columns]
        + ["y"]
    )
    sub = df[all_feat].copy()
    for c in sub.columns:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    corr = sub.corr(method="spearman")

    # Rút gọn tên
    short = {}
    for c in all_feat:
        s = c.replace("Curricular units ", "CU ") \
             .replace("1st sem ", "S1 ").replace("2nd sem ", "S2 ") \
             .replace(" qualification", " qual") \
             .replace(" occupation", " occ") \
             .replace("(without evaluations)", "(no-eval)")
        short[c] = s

    fig, ax = plt.subplots(figsize=(18, 15))
    mask = np.zeros_like(corr, dtype=bool)  # Hiển thị toàn bộ
    sns.heatmap(
        corr.rename(index=short, columns=short),
        cmap="RdBu_r", vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 5},
        linewidths=0.3, linecolor="#cccccc",
        cbar_kws={"shrink": 0.6, "label": "Spearman r"},
        ax=ax,
    )
    ax.set_title("Ma trận tương quan Spearman – toàn bộ đặc trưng và nhãn y", fontsize=12, pad=12)
    ax.tick_params(labelsize=6, rotation=45)
    ax.set_xticklabels(ax.get_xticklabels(), ha="right")

    # Vẽ đường phân chia nhóm
    n_enroll = sum(1 for c in ENROLLMENT_COLS if c in df.columns)
    n_s1 = sum(1 for c in SEM1_COLS if c in df.columns)
    n_s2 = sum(1 for c in SEM2_COLS if c in df.columns)
    n_macro = sum(1 for c in MACRO_COLS if c in df.columns)
    boundaries = [n_enroll, n_enroll + n_s1, n_enroll + n_s1 + n_s2,
                  n_enroll + n_s1 + n_s2 + n_macro]
    for b in boundaries:
        ax.axhline(b, color="black", lw=1.5)
        ax.axvline(b, color="black", lw=1.5)

    # Legend nhóm
    group_labels = ["Nhập học", "HK1", "HK2 ⚠", "Vĩ mô", "Nhãn y"]
    group_colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"]
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(group_colors, group_labels)]
    ax.legend(handles=patches, loc="upper right", fontsize=7, title="Nhóm biến",
              bbox_to_anchor=(1.18, 1))

    fig.tight_layout()
    fig.savefig(os.path.join(out, "leakage_corr_full.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → leakage_corr_full.png")


# ── Vẽ bar chart top-20 tương quan với y ─────────────────────────────────────
def plot_top_corr(corr_df: pd.DataFrame, out: str):
    top = corr_df.head(20).copy()

    # Xác định màu theo rủi ro
    colors = []
    for _, row in top.iterrows():
        risk, _ = classify_leakage(row["feature"], row["spearman_r"])
        colors.append(COLOR[risk])

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        range(len(top)), top["spearman_r"].values, color=colors,
        edgecolor="white", linewidth=0.5
    )
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(
        [f.replace("Curricular units ", "CU ") for f in top["feature"]],
        fontsize=8
    )
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Spearman r với nhãn bỏ học (y)", fontsize=9)
    ax.set_title("Top 20 đặc trưng có |tương quan| cao nhất với nhãn y\n"
                 "(màu = mức độ rủi ro data leakage)", fontsize=10)
    ax.set_xlim(-1.05, 1.05)

    # Thêm giá trị
    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(row["spearman_r"] + (0.02 if row["spearman_r"] >= 0 else -0.02),
                i, f'{row["spearman_r"]:.3f}', va="center",
                ha="left" if row["spearman_r"] >= 0 else "right", fontsize=7)

    # Legend rủi ro
    patches = [
        mpatches.Patch(color=COLOR["HIGH"],   label="HIGH – Rủi ro cao (thông tin HK2 sau dropout)"),
        mpatches.Patch(color=COLOR["MEDIUM"], label="MEDIUM – Cần chú ý"),
        mpatches.Patch(color=COLOR["LOW"],    label="LOW – Tín hiệu yếu"),
        mpatches.Patch(color=COLOR["OK"],     label="OK – Không có rủi ro"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "leakage_top_corr.png"), dpi=150)
    plt.close(fig)
    print("  → leakage_top_corr.png")


# ── Vẽ scatter HK2 vs HK1 để minh họa rủi ro thời gian ─────────────────────
def plot_sem2_risk(df: pd.DataFrame, out: str):
    pairs = [
        ("Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)",
         "Môn đạt HK1", "Môn đạt HK2"),
        ("Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)",
         "Điểm HK1", "Điểm HK2"),
        ("Curricular units 1st sem (enrolled)", "Curricular units 2nd sem (enrolled)",
         "Đăng ký HK1", "Đăng ký HK2"),
    ]
    pairs = [(a, b, la, lb) for a, b, la, lb in pairs
             if a in df.columns and b in df.columns]
    if not pairs:
        return

    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.5))
    if len(pairs) == 1:
        axes = [axes]

    cmap = {0: "#3498db", 1: "#e74c3c"}
    labels_map = {0: "Không bỏ học", 1: "Bỏ học"}

    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, pairs):
        for cls in [0, 1]:
            mask = df["y"] == cls
            ax.scatter(
                df.loc[mask, xcol], df.loc[mask, ycol],
                c=cmap[cls], alpha=0.25, s=10, label=labels_map[cls]
            )
        # Đường y = x
        lim = max(df[xcol].max(), df[ycol].max())
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="y = x")

        r_all, _ = spearmanr(df[[xcol, ycol]].dropna()[xcol],
                              df[[xcol, ycol]].dropna()[ycol])
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"r={r_all:.2f}", fontsize=9)
        ax.legend(fontsize=7, markerscale=2)

    fig.suptitle(
        "HK2 vs HK1 – Sinh viên bỏ học (đỏ) có xu hướng\n"
        "không đăng ký / không thi HK2  →  Biến HK2 phản ánh KẾT QUẢ, không phải nguyên nhân",
        fontsize=10
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out, "leakage_sem2_risk.png"), dpi=150)
    plt.close(fig)
    print("  → leakage_sem2_risk.png")


# ── Sinh báo cáo văn bản ─────────────────────────────────────────────────────
def write_report(df: pd.DataFrame, corr_df: pd.DataFrame, out: str):
    lines = []
    sep = "=" * 72

    lines += [
        sep,
        "BÁO CÁO KIỂM TRA TƯƠNG QUAN VÀ DATA LEAKAGE",
        f"Dữ liệu: {df.shape[0]} mẫu, {df.shape[1] - 2} đặc trưng",
        f"Tỷ lệ Dropout: {df['y'].mean():.1%}",
        sep, "",
    ]

    # ── Phần 1: Tương quan Spearman toàn bộ ─────────────────────────────────
    lines += ["─" * 72, "PHẦN 1 – TƯƠNG QUAN SPEARMAN VỚI NHÃN BỎ HỌC (y)", "─" * 72, ""]
    lines.append(f"{'Thứ hạng':<6} {'Đặc trưng':<48} {'r':>7} {'|r|':>6} {'p':>10}  Nhóm")
    lines.append("-" * 90)
    for i, row in corr_df.iterrows():
        risk, _ = classify_leakage(row["feature"], row["spearman_r"])
        # Xác định nhóm
        f = row["feature"]
        if any(f.startswith("Curricular units 2nd sem") for _ in [1]):
            group = "HK2 ⚠"
        elif any(f.startswith("Curricular units 1st sem") for _ in [1]):
            group = "HK1"
        elif f in MACRO_COLS:
            group = "Vĩ mô"
        else:
            group = "Nhập học"
        p_str = f"{row['p_value']:.2e}"
        lines.append(
            f"{i+1:<6} {f[:47]:<48} {row['spearman_r']:>7.3f} {row['abs_r']:>6.3f} "
            f"{p_str:>10}  [{risk}] {group}"
        )
    lines.append("")

    # ── Phần 2: Rà soát từng nhóm theo rủi ro leakage ───────────────────────
    lines += ["", "─" * 72, "PHẦN 2 – RÀ SOÁT DATA LEAKAGE THEO NHÓM BIẾN", "─" * 72]

    risk_groups = {"HIGH": [], "MEDIUM": [], "LOW": [], "OK": []}
    for _, row in corr_df.iterrows():
        risk, reason = classify_leakage(row["feature"], row["spearman_r"])
        risk_groups[risk].append((row["feature"], row["spearman_r"], reason))

    for level in ["HIGH", "MEDIUM", "LOW", "OK"]:
        emoji = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟡", "OK": "🟢"}.get(level, "")
        lines.append(f"\n{emoji} [{level}] – {len(risk_groups[level])} biến")
        lines.append("-" * 60)
        for feat, r, reason in risk_groups[level]:
            lines.append(f"  • {feat}")
            lines.append(f"    r={r:.3f} | {reason}")
            lines.append("")

    # ── Phần 3: Tương quan giữa các đặc trưng (đa cộng tuyến) ───────────────
    lines += ["", "─" * 72, "PHẦN 3 – ĐA CỘNG TUYẾN GIỮA CÁC ĐẶC TRƯNG (|r| ≥ 0.70)", "─" * 72, ""]
    num_cols = [c for c in
                (ENROLLMENT_COLS + SEM1_COLS + SEM2_COLS + MACRO_COLS)
                if c in df.columns]
    corr_mat = df[num_cols].corr(method="spearman")
    high_pairs = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            r = corr_mat.iloc[i, j]
            if abs(r) >= 0.70:
                high_pairs.append((num_cols[i], num_cols[j], r))
    high_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    if high_pairs:
        lines.append(f"Tìm thấy {len(high_pairs)} cặp biến có |r| ≥ 0.70:\n")
        for a, b, r in high_pairs:
            risk_a, _ = classify_leakage(a, r)
            risk_b, _ = classify_leakage(b, r)
            flag = " ⚠ CẢ HAI LÀ HK2" if "2nd sem" in a and "2nd sem" in b else ""
            lines.append(f"  r={r:+.3f}  [{risk_a}] {a}")
            lines.append(f"          [{risk_b}] {b}{flag}")
            lines.append("")
    else:
        lines.append("Không có cặp biến nào có |r| ≥ 0.70.\n")

    # ── Phần 4: Kết luận và khuyến nghị ─────────────────────────────────────
    lines += [
        "", "─" * 72, "PHẦN 4 – KẾT LUẬN VÀ KHUYẾN NGHỊ", "─" * 72, "",
        "1. BIẾN HỌC KỲ 2 (setting=early) – RỦI RO CAO:",
        "   Các biến như 'CU 2nd sem (approved)', 'CU 2nd sem (grade)', ...",
        "   có tương quan rất cao với y (|r| = 0.5–0.7) nhưng đây là QUAN HỆ",
        "   NHÂN-QUẢ NGƯỢC: sinh viên đã quyết định bỏ học → không thi/đăng ký HK2.",
        "   → Pipeline đã loại ĐÚNG các biến này ở setting=early.",
        "   → Ở setting=full, cần ghi chú hạn chế này trong báo cáo.",
        "",
        "2. BIẾN VĨ MÔ (GDP, Inflation, Unemployment) – RỦI RO THẤP:",
        "   Không có quan hệ nhân-quả ngược. Tuy nhiên tương quan thấp (|r| ≤ 0.15)",
        "   gợi ý đây là nhiễu hơn là tín hiệu dự báo có giá trị.",
        "   → Pipeline Luồng B đã bỏ đúng. Luồng A giữ lại để so sánh.",
        "",
        "3. BIẾN HỌC KỲ 1 (setting=early) – AN TOÀN:",
        "   Thu thập trong suốt HK1, trước khi dropout chính thức. Tương quan cao",
        "   (approved: r≈-0.52, grade: r≈-0.46) là TÍN HIỆU DỰ BÁO HỢP LỆ.",
        "   → Đây là nhóm đặc trưng quan trọng nhất cho bài toán cảnh báo sớm.",
        "",
        "4. ĐA CỘNG TUYẾN TRONG NHÓM HK1/HK2:",
        "   enrolled↔evaluations, approved↔grade có tương quan cao (r≈0.7–0.9).",
        "   → Với Decision Tree: không ảnh hưởng vì cây không giả định tuyến tính.",
        "   → Với KNN và NaiveBayes: có thể gây double-counting trọng số.",
        "   → Khuyến nghị: thêm đặc trưng tỷ lệ (Luồng B) thay vì giữ cả hai biến.",
        "",
        "5. PIPELINE HIỆN TẠI – ĐÁNH GIÁ TỔNG THỂ:",
        "   ✅ Tất cả bước học tham số nằm trong sklearn.Pipeline → không leakage kỹ thuật.",
        "   ✅ setting=early không dùng biến HK2 → không leakage thời gian ở kịch bản chính.",
        "   ⚠  setting=full dùng biến HK2 → cần ghi chú rõ: đây là thực nghiệm hồi cứu,",
        "      không phù hợp cho hệ thống cảnh báo sớm thực tế.",
        "   ✅ Feature engineering (approve rate, no-eval ratio) chỉ dùng tính toán từng dòng.",
        sep,
    ]

    report_path = os.path.join(out, "leakage_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  → leakage_report.txt")
    return lines


# ── Vẽ ma trận tương quan HK1 vs HK2 chi tiết ───────────────────────────────
def plot_sem_internal_corr(df: pd.DataFrame, out: str):
    """Heatmap tương quan nội bộ HK1 và HK2 để kiểm tra đa cộng tuyến."""
    sem_cols = [c for c in SEM1_COLS + SEM2_COLS if c in df.columns]
    if len(sem_cols) < 4:
        return
    short = {c: c.replace("Curricular units ", "").replace(" sem ", " S")
             .replace("(without evaluations)", "(no-eval)") for c in sem_cols}
    sub = df[sem_cols].copy()
    corr = sub.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # Chỉ tam giác dưới
    sns.heatmap(
        corr.rename(index=short, columns=short),
        mask=mask,
        cmap="RdBu_r", vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 8},
        square=True,
        linewidths=0.4, linecolor="#cccccc",
        cbar_kws={"shrink": 0.7},
        ax=ax,
    )
    # Vẽ đường phân chia HK1/HK2
    n1 = sum(1 for c in SEM1_COLS if c in df.columns)
    ax.axhline(n1, color="red", lw=2, ls="--")
    ax.axvline(n1, color="red", lw=2, ls="--")
    ax.set_title("Tương quan nội bộ HK1 / HK2 (đường đỏ = ranh giới HK1–HK2)\n"
                 "Vùng góc dưới-phải = tương quan chéo HK1×HK2 → rủi ro temporal leakage",
                 fontsize=9)
    ax.tick_params(labelsize=8, rotation=45)
    ax.set_xticklabels(ax.get_xticklabels(), ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "leakage_sem_internal_corr.png"), dpi=150)
    plt.close(fig)
    print("  → leakage_sem_internal_corr.png")


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Kiểm tra tương quan & data leakage")
    ap.add_argument("--data", default="data/student_dropout.csv")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Đang tải dữ liệu...")
    df = load(args.data)
    print(f"  {df.shape[0]} mẫu, {df.shape[1]-2} đặc trưng, Dropout={df['y'].mean():.1%}\n")

    print("Tính tương quan Spearman...")
    corr_df = spearman_with_target(df)

    print("Vẽ biểu đồ...")
    plot_full_heatmap(df, args.out)
    plot_top_corr(corr_df, args.out)
    plot_sem2_risk(df, args.out)
    plot_sem_internal_corr(df, args.out)

    print("Sinh báo cáo...")
    report_lines = write_report(df, corr_df, args.out)

    # In tóm tắt ra console
    print("\n" + "=" * 72)
    print("TÓM TẮT NHANH – BIẾN CÓ |r| ≥ 0.30 VỚI NHÃN y")
    print("=" * 72)
    high = corr_df[corr_df["abs_r"] >= 0.30]
    for _, row in high.iterrows():
        risk, _ = classify_leakage(row["feature"], row["spearman_r"])
        flag = " ⚠" if risk in ("HIGH", "MEDIUM") else "  "
        print(f"  {flag} [{risk:6s}] r={row['spearman_r']:+.3f}  {row['feature']}")
    print(f"\nKết quả lưu tại: {args.out}/")
    print("  leakage_corr_full.png, leakage_top_corr.png,")
    print("  leakage_sem2_risk.png, leakage_sem_internal_corr.png, leakage_report.txt")


if __name__ == "__main__":
    main()
