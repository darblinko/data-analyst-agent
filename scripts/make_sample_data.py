"""Write the synthetic sample dataset used by demo mode and the tests.

The data is generated from a fixed seed, so rerunning this script reproduces the
committed file byte for byte. No network access and no real business data.

Usage: uv run python scripts/make_sample_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261029
OUTPUT = Path(__file__).resolve().parents[1] / "src/data_analyst_agent/sample_data/store_sales.csv"
REGION_SCALE = {"North": 1.0, "South": 0.8, "East": 1.2, "West": 0.6}
REGION_ANNUAL_GROWTH = {"North": 0.03, "South": 0.08, "East": -0.02, "West": 0.22}
CATEGORY_UNITS = {"Hardware": 40, "Software": 25, "Services": 12}
CATEGORY_PRICE = {"Hardware": 120.0, "Software": 80.0, "Services": 150.0}
PRICE_INCREASE_2025 = 1.04
MISSING_ROWS = 12


def build_frame(rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    grid = pd.MultiIndex.from_product(
        [dates, list(REGION_SCALE), list(CATEGORY_UNITS)], names=["date", "region", "category"]
    ).to_frame(index=False)
    years = (grid["date"] - dates[0]).dt.days / 365.25
    growth = np.exp(np.log1p(grid["region"].map(REGION_ANNUAL_GROWTH)) * years)
    day_of_year = grid["date"].dt.dayofyear
    annual_season = 1 + 0.15 * np.cos(2 * np.pi * (day_of_year - 350) / 365.25)
    weekend = np.where(grid["date"].dt.dayofweek >= 5, 0.7, 1.0)
    expected_units = (
        grid["category"].map(CATEGORY_UNITS)
        * grid["region"].map(REGION_SCALE)
        * growth
        * annual_season
        * weekend
    )
    units = rng.poisson(expected_units.to_numpy()).astype(float)
    price_level = np.where(grid["date"].dt.year >= 2025, PRICE_INCREASE_2025, 1.0)
    price_noise = rng.normal(1.0, 0.03, len(grid))
    unit_price = (grid["category"].map(CATEGORY_PRICE) * price_level * price_noise).round(2)
    missing = rng.choice(len(grid), size=MISSING_ROWS, replace=False)
    units[missing] = np.nan
    return grid.assign(
        units=pd.array(units, dtype="Int64"),
        unit_price=unit_price,
        revenue=(units * unit_price).round(2),
    )


def main() -> None:
    frame = build_frame(np.random.default_rng(SEED))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False, date_format="%Y-%m-%d")
    print(f"wrote {len(frame):,} rows to {OUTPUT.name}")


if __name__ == "__main__":
    main()
