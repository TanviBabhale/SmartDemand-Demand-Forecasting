import pandas as pd


def load_dataset(file_path):
    """Load a CSV dataset."""

    df = pd.read_csv(file_path)

    print("===== DATASET LOADED =====")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    return df


def detect_date_columns(df):
    """Find columns that are likely to contain dates."""

    possible_dates = []

    for column in df.columns:

        # Skip numeric columns
        if pd.api.types.is_numeric_dtype(df[column]):
            continue

        # Skip boolean columns
        if pd.api.types.is_bool_dtype(df[column]):
            continue

        converted = pd.to_datetime(
            df[column],
            errors="coerce"
        )

        valid_ratio = converted.notna().mean()

        if valid_ratio >= 0.70:
            possible_dates.append(column)

    return possible_dates


def detect_numeric_columns(df):
    """Find numeric columns suitable for forecasting."""

    numeric_columns = df.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    return numeric_columns


def show_dataset_information(df):

    print("\n===== DATASET INFORMATION =====")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nPossible date columns:")
    print(detect_date_columns(df))

    print("\nNumeric columns:")
    print(detect_numeric_columns(df))


def suggest_target_columns(df):
    """Suggest numeric columns that could be forecast targets."""

    numeric_columns = detect_numeric_columns(df)

    excluded_keywords = [
        "id",
        "code",
        "store",
        "dept",
        "department",
        "year",
        "month",
        "day"
    ]

    candidates = []

    for column in numeric_columns:

        column_lower = column.lower()

        if any(
            keyword in column_lower
            for keyword in excluded_keywords
        ):
            continue

        candidates.append(column)

    return candidates