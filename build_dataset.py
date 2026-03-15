from pathlib import Path

import pandas as pd


def build_dataset(base_dir: Path) -> pd.DataFrame:
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    crunchbase = pd.read_csv(raw_dir / "crunchbase.csv")
    angellist = pd.read_csv(raw_dir / "angellist.csv")
    pitchbook = pd.read_csv(raw_dir / "pitchbook.csv")

    df = crunchbase.merge(
        angellist,
        on=["startup_id", "startup_name"],
        how="left",
    ).merge(
        pitchbook,
        on=["startup_id", "startup_name"],
        how="left",
    )

    current_year = 2026
    df["startup_age_years"] = current_year - df["founded_year"]

    numeric_columns = [
        "funding_rounds",
        "total_funding_usd",
        "employees",
        "team_size",
        "followers",
        "job_openings",
        "monthly_web_visits",
        "investor_count",
        "latest_valuation_usd",
        "has_exit",
        "years_to_exit",
        "startup_age_years",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    output_file = processed_dir / "startup_dataset.csv"
    df.to_csv(output_file, index=False)
    return df


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    dataset = build_dataset(base_dir)

    print("Dataset built successfully: data/processed/startup_dataset.csv")
    print("\nShape:")
    print(dataset.shape)

    print("\nHead:")
    print(dataset.head())

    print("\nTail:")
    print(dataset.tail())

    print("\nDescribe (numeric):")
    print(dataset.describe())


if __name__ == "__main__":
    main()
