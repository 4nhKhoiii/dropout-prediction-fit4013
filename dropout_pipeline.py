import argparse
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn import __version__ as SKLEARN_VERSION  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Khai báo nhóm cột (tên đã được strip khoảng trắng và sửa lỗi "Nacionality")
# ---------------------------------------------------------------------------
CAT_COLS = [
    "Marital status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nationality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]
BIN_COLS = [
    "Daytime/evening attendance",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]
BASE_NUM = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Application order",
]
MACRO_COLS = ["Unemployment rate", "Inflation rate", "GDP"]

SEM_FIELDS = ["(credited)", "(enrolled)", "(evaluations)", "(approved)", "(grade)", "(without evaluations)"]
SEM1_COLS = [f"Curricular units 1st sem {f}" for f in SEM_FIELDS]
SEM2_COLS = [f"Curricular units 2nd sem {f}" for f in SEM_FIELDS]
EARLY_INPUT_COLS = BASE_NUM + BIN_COLS + CAT_COLS + SEM1_COLS


# ---------------------------------------------------------------------------
# 1. Nạp và làm sạch dữ liệu
# ---------------------------------------------------------------------------
def normalize_missing_grades(df, setting):
    """Treat a semester grade as missing when the student had no evaluations."""
    semesters = ["1st sem"] + (["2nd sem"] if setting == "full" else [])
    for semester in semesters:
        eval_col = f"Curricular units {semester} (evaluations)"
        grade_col = f"Curricular units {semester} (grade)"
        df.loc[df[eval_col] == 0, grade_col] = np.nan
    return df


def load_data(path):
    """Đọc CSV tự nhận diện dấu phân cách và chuẩn hóa tên cột."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {path}. "
            "Hãy kiểm tra --data hoặc đặt CSV đúng thư mục."
        )

    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [
        str(c).strip().replace("\ufeff", "").replace("Nacionality", "Nationality")
        for c in df.columns
    ]

    # Chuẩn hóa tên cột theo danh sách tên chuẩn, không phân biệt hoa/thường.
    # Ví dụ: "Marital Status" -> "Marital status".
    expected_names = (
        CAT_COLS + BIN_COLS + BASE_NUM + MACRO_COLS + SEM1_COLS + SEM2_COLS
        + ["Target"]
    )
    canonical_names = {name.casefold(): name for name in expected_names}
    rename_map = {
        col: canonical_names[col.casefold()]
        for col in df.columns
        if col.casefold() in canonical_names and col != canonical_names[col.casefold()]
    }
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # Chuẩn hóa tên cột Target, không phân biệt chữ hoa/thường.
    target_col = next((c for c in df.columns if c.lower() == "target"), None)
    if target_col is None:
        raise ValueError(
            "Không tìm thấy cột Target. Có thể CSV đang dùng sai dấu phân cách "
            f"hoặc tên cột khác. Các cột đọc được: {df.columns.tolist()}"
        )
    if target_col != "Target":
        df.rename(columns={target_col: "Target"}, inplace=True)

    # Kiểm tra các cột mà mô hình bắt buộc phải sử dụng.
    required = (
        CAT_COLS + BIN_COLS + BASE_NUM + MACRO_COLS + SEM1_COLS + SEM2_COLS
        + ["Target"]
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Dataset thiếu các cột cần cho pipeline:\n- "
            + "\n- ".join(missing)
            + f"\nCác cột hiện có: {df.columns.tolist()}"
        )

    df["Target"] = df["Target"].astype(str).str.strip()
    valid_targets = {"Graduate", "Enrolled", "Dropout"}
    unexpected = sorted(set(df["Target"].dropna()) - valid_targets)
    if unexpected:
        raise ValueError(f"Giá trị Target không đúng định dạng dự kiến: {unexpected}")

    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Đã loại {n_dup} dòng trùng lặp. Kích thước sau làm sạch: {df.shape}")
    print("Phân bố Target:\\n", df["Target"].value_counts().to_string())

    # Ép các cột số về numeric; giá trị không hợp lệ được chuyển thành NaN
    numeric_cols = BIN_COLS + BASE_NUM + MACRO_COLS + SEM1_COLS + SEM2_COLS
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Khi không có đánh giá nào (evaluations == 0), điểm 0.0 được xem là không có điểm.
    normalize_missing_grades(df, "full")

    df["y"] = (df["Target"].str.lower() == "dropout").astype(int)
    if df["y"].nunique() < 2:
        raise ValueError("Target chỉ có một lớp sau khi đọc dữ liệu; không thể huấn luyện mô hình.")
    return df


# ---------------------------------------------------------------------------
# 2. Khám phá dữ liệu (EDA)
# ---------------------------------------------------------------------------
def eda(df, out):
    print("\n== EDA ==")
    print("Kích thước:", df.shape)
    print("Phân phối Target:\n", df["Target"].value_counts(), sep="")
    print("Tỷ lệ Dropout:", round(df["y"].mean(), 3))
    print("Số ô thiếu (NaN sau xử lý điểm 0):", int(df.isna().sum().sum()))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    df["Target"].value_counts().plot.bar(ax=ax[0], title="Phân phối nhãn")
    for lab in ["Graduate", "Enrolled", "Dropout"]:
        ax[1].hist(df.loc[df["Target"] == lab, "Admission grade"], bins=30, alpha=0.5, label=lab)
    ax[1].set_title("Admission grade theo nhãn")
    ax[1].legend()
    df.boxplot(column="Curricular units 1st sem (approved)", by="Target", ax=ax[2])
    ax[2].set_title("Số môn đạt HK1 theo nhãn")
    ax[2].set_xlabel("")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "eda_overview.png"), dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# 3. Đặc trưng: kỹ thuật đặc trưng (chỉ biến đổi từng dòng, không học tham số)
# ---------------------------------------------------------------------------
def engineer(df, setting):
    """Thêm đặc trưng mới. Vì chỉ tính trên từng dòng nên không gây data leakage."""
    d = df.copy()
    sems = ["1st sem"] + (["2nd sem"] if setting == "full" else [])
    for s in sems:
        p = f"Curricular units {s}"
        enr = d[f"{p} (enrolled)"].replace(0, np.nan)
        d[f"{s} approval rate"] = (d[f"{p} (approved)"] / enr).fillna(0)
        d[f"{s} no-eval ratio"] = (d[f"{p} (without evaluations)"] / enr).fillna(0)
    if setting == "full":
        d["grade change"] = d["Curricular units 2nd sem (grade)"] - d["Curricular units 1st sem (grade)"]
    return d


def feature_sets(setting, flow):
    """Trả về (danh sách cột số, danh sách cột phân loại) theo phạm vi và luồng."""
    sems = SEM1_COLS + (SEM2_COLS if setting == "full" else [])
    num = BASE_NUM + BIN_COLS + sems
    if flow == "A":
        # Luồng A: pipeline cơ bản, giữ nguyên dữ liệu thô + nhóm biến vĩ mô
        num = num + MACRO_COLS
    else:
        # Luồng B: thêm đặc trưng tỷ lệ, bỏ 3 biến vĩ mô (cấp vùng, tín hiệu yếu, dễ học nhiễu)
        eng = ["1st sem approval rate", "1st sem no-eval ratio"]
        if setting == "full":
            eng += ["2nd sem approval rate", "2nd sem no-eval ratio", "grade change"]
        num = num + eng
    return num, CAT_COLS


def make_preprocessor(num_cols, cat_cols, scale):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


# ---------------------------------------------------------------------------
# 4. Mô hình và lưới siêu tham số
# ---------------------------------------------------------------------------
def model_specs():
    """(tên, estimator, có chuẩn hóa không, lưới siêu tham số hoặc None)"""
    return [
        ("Baseline_Majority", DummyClassifier(strategy="most_frequent"), False, None),
        ("DecisionTree_default", DecisionTreeClassifier(random_state=RANDOM_STATE), False, None),
        ("KNN", KNeighborsClassifier(), True, {
            "clf__n_neighbors": [3, 5, 9, 15, 21],
            "clf__weights": ["uniform", "distance"],
        }),
        ("NaiveBayes", GaussianNB(), True, {
            "clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3],
        }),
        ("DecisionTree", DecisionTreeClassifier(random_state=RANDOM_STATE), False, {
            "clf__max_depth": [3, 5, 7, 10, None],
            "clf__min_samples_leaf": [5, 10, 20, 50],
            "clf__criterion": ["gini", "entropy"],
            "clf__class_weight": [None, "balanced"],
        }),
    ]


def fit_one(est, scale, grid, num, cat, X_tr, y_tr, cv):
    pipe = Pipeline([("prep", make_preprocessor(num, cat, scale)), ("clf", est)])
    if grid:
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        return gs.best_estimator_, gs.best_params_, gs.best_score_
    pipe.fit(X_tr, y_tr)
    cv_f1 = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1").mean()
    return pipe, {}, cv_f1


def test_metrics(est, X_te, y_te):
    y_pred = est.predict(X_te)
    proba = est.predict_proba(X_te)[:, 1]
    metrics = {
        "test_accuracy": accuracy_score(y_te, y_pred),
        "test_precision": precision_score(y_te, y_pred, zero_division=0),
        "test_recall": recall_score(y_te, y_pred, zero_division=0),
        "test_f1": f1_score(y_te, y_pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_te, proba),
    }
    return metrics, y_pred


# ---------------------------------------------------------------------------
# 5. Thí nghiệm
# ---------------------------------------------------------------------------
def save_confusion(y_te, y_pred, path, title):
    cm = confusion_matrix(y_te, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Không bỏ học", "Bỏ học"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def save_tree_importance(pipe, path, title):
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(pipe.named_steps["clf"].feature_importances_, index=names)
    imp = imp.sort_values(ascending=False).head(15)[::-1]
    imp.plot.barh(figsize=(8, 6), title=title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def run_experiments(df, out):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted = [], {}

    for setting in ["early", "full"]:
        data = engineer(df, setting)
        for flow in ["A", "B"]:
            num, cat = feature_sets(setting, flow)
            X = data[num + cat]
            y = data["y"]
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
            )
            print(f"\n--- setting={setting} | flow={flow} | train={X_tr.shape} test={X_te.shape}")

            for name, est, scale, grid in model_specs():
                best, params, cv_f1 = fit_one(est, scale, grid, num, cat, X_tr, y_tr, cv)
                m, y_pred = test_metrics(best, X_te, y_te)
                key = (setting, flow, name)
                fitted[key] = (best, X_te, y_te, y_pred)

                row = {"setting": setting, "flow": flow, "model": name,
                       "best_params": str(params), "cv_f1": cv_f1}
                row.update(m)
                rows.append(row)
                print(f"  {name:22s} CV-F1={cv_f1:.3f}  test-F1={m['test_f1']:.3f}  "
                      f"recall={m['test_recall']:.3f}  AUC={m['test_roc_auc']:.3f}")

                tag = f"{setting}_{flow}_{name}"
                save_confusion(y_te, y_pred, os.path.join(out, f"cm_{tag}.png"), tag)
                if name == "DecisionTree":
                    save_tree_importance(best, os.path.join(out, f"importance_{tag}.png"), tag)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "metrics_all.csv"), index=False)
    print("\n== Bảng kết quả (sắp theo CV-F1) ==")
    print(res.sort_values("cv_f1", ascending=False).round(3).to_string(index=False))

    print("\n== Test F1 theo mô hình và phương án ==")
    print(res.pivot_table(index="model", columns=["setting", "flow"], values="test_f1").round(3))
    return res, fitted


def save_best_early_model(res, fitted, model_dir):
    """Persist the best fitted early pipeline, selected only by cross-validation F1."""
    candidates = res[
        (res["setting"] == "early")
        & (~res["model"].str.startswith("Baseline"))
        & res["cv_f1"].notna()
    ]
    if candidates.empty:
        raise ValueError("Không có kết quả CV hợp lệ để chọn mô hình early.")

    best_row = candidates.sort_values("cv_f1", ascending=False).iloc[0]
    key = (best_row["setting"], best_row["flow"], best_row["model"])
    if key not in fitted:
        raise ValueError(f"Không tìm thấy mô hình đã fit tương ứng với kết quả tốt nhất: {key}")

    estimator = fitted[key][0]
    num_cols, cat_cols = feature_sets(best_row["setting"], best_row["flow"])
    artifact = {
        "model": estimator,
        "metadata": {
            "setting": best_row["setting"],
            "flow": best_row["flow"],
            "model_name": best_row["model"],
            "cv_f1": float(best_row["cv_f1"]),
            "best_params": best_row["best_params"],
            "input_columns": EARLY_INPUT_COLS,
            "feature_columns": num_cols + cat_cols,
            "sklearn_version": SKLEARN_VERSION,
        },
    }

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "best_early_model.joblib")
    joblib.dump(artifact, model_path)
    return model_path


# ---------------------------------------------------------------------------
# 6. Phân tích lỗi cho mô hình tốt nhất (chọn theo CV-F1, không dùng test)
# ---------------------------------------------------------------------------
def error_analysis(X_te, y_te, y_pred, out):
    e = X_te.copy()
    e["y_true"] = y_te.values
    e["y_pred"] = y_pred
    e["group"] = np.select(
        [
            (e.y_true == 1) & (e.y_pred == 1),
            (e.y_true == 0) & (e.y_pred == 0),
            (e.y_true == 1) & (e.y_pred == 0),
        ],
        ["TP", "TN", "FN"],
        default="FP",
    )
    print("\nSố lượng theo nhóm:")
    print(e["group"].value_counts())

    cols = [c for c in [
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Admission grade",
        "Age at enrollment",
        "Tuition fees up to date",
        "Debtor",
        "Scholarship holder",
    ] if c in e.columns]

    profile = e.groupby("group")[cols].mean().round(2)
    profile.to_csv(os.path.join(out, "error_profile.csv"))
    print("\n== Đặc điểm trung bình theo nhóm dự đoán ==")
    print(profile)

    print("\nTop 10 khóa học có nhiều False Negative (bỏ học nhưng dự đoán không):")
    print(e.loc[e.group == "FN", "Course"].value_counts().head(10))

    e[e.group.isin(["FN", "FP"])].to_csv(os.path.join(out, "misclassified.csv"), index=False)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/student_dropout.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--model-out", default="models")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    np.random.seed(RANDOM_STATE)
    df = load_data(args.data)
    eda(df, args.out)

    res, fitted = run_experiments(df, args.out)

    model_path = save_best_early_model(res, fitted, args.model_out)
    print(f"\nĐã lưu mô hình early tốt nhất (chọn theo CV-F1): {model_path}")

    # Phân tích lỗi và lưu mô hình đều chọn theo CV-F1, không dùng tập kiểm thử.
    cand = res[(res["setting"] == "early") & (~res["model"].str.startswith("Baseline"))]
    best_row = cand.sort_values("cv_f1", ascending=False).iloc[0]
    key = ("early", best_row["flow"], best_row["model"])
    print(f"\nMô hình được chọn để phân tích lỗi (early, theo CV-F1): {key}")

    _, X_te, y_te, y_pred = fitted[key]
    error_analysis(X_te, y_te, y_pred, args.out)
    print(f"\nHoàn tất. Kết quả lưu tại thư mục: {args.out}")


if __name__ == "__main__":
    main()