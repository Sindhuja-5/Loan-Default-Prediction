import pandas as pd

from sklearn.preprocessing import LabelEncoder


def preprocess_data(df):
    # ============================================================
    # LOAD DATA
    # ============================================================
    df = df.drop_duplicates()


    # ============================================================
    # DROP UNNECESSARY COLUMNS
    # ============================================================

    columns_to_drop = [
        "desc",
        "mths_since_last_delinq",
        "mths_since_last_record",
        "mths_since_last_major_derog",
        "verification_status_joint"
    ]

    df = df.drop(columns=columns_to_drop)

    df.drop(
        ["zip_code", "title", "member_id"],
        axis=1,
        inplace=True
    )


    # ============================================================
    # FEATURE ENGINEERING
    # ============================================================

    # Loan term
    df["term"] = df["term"].str.replace(
        " months",
        "",
        regex=False
    )

    df["term"] = pd.to_numeric(
        df["term"],
        errors="coerce"
    )


    # Employment length
    df["emp_length"] = df["emp_length"].replace(
        "NAN",
        "0"
    )

    df["emp_length"] = df["emp_length"].str.replace(
        r"\+ years",
        "",
        regex=True
    )

    df["emp_length"] = df["emp_length"].str.replace(
        " years",
        "",
        regex=False
    )

    df["emp_length"] = df["emp_length"].replace(
        "< 1 year",
        "0"
    )

    df["emp_length"] = df["emp_length"].str.replace(
        " year",
        "",
        regex=False
    )

    df["emp_length"] = pd.to_numeric(
        df["emp_length"],
        errors="coerce"
    )


    # Last week pay
    df["last_week_pay"] = df["last_week_pay"].str.replace(
        "th week",
        "",
        regex=False
    )

    df["last_week_pay"] = df["last_week_pay"].replace(
        "NA",
        ""
    )

    df["last_week_pay"] = pd.to_numeric(
        df["last_week_pay"],
        errors="coerce"
    )


    # ============================================================
    # MISSING VALUES
    # ============================================================

    numeric_cols = df.select_dtypes(
        include="number"
    ).columns

    df[numeric_cols] = df[numeric_cols].apply(
        lambda x: x.fillna(x.median())
    )

    df["emp_title"] = df["emp_title"].fillna("Other")


    # ============================================================
    # CONVERT GRADES
    # ============================================================

    grade_mapping = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
        "G": 6
    }

    for column in ["sub_grade", "grade"]:

        df[column] = df[column].replace(
            grade_mapping
        )

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    # ============================================================
    # BATCH ENROLLED
    # ============================================================

    df["batch_enrolled"] = df["batch_enrolled"].str.replace(
        "^BAT",
        "",
        regex=True
    )

    df["batch_enrolled"] = pd.to_numeric(
        df["batch_enrolled"],
        errors="coerce"
    )

    df["batch_enrolled"] = df["batch_enrolled"].fillna(
        df["batch_enrolled"].median()
    )


    # ============================================================
    # CATEGORICAL ENCODING
    # ============================================================

    categorical_cols = [
        col for col in df.columns
        if df[col].dtype == "object"
    ]

    for col in categorical_cols:

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col])

    return df