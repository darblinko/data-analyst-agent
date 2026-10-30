"""Scripted questions for demo mode, written against the bundled sample dataset.

The final answers quote numbers that the scripted code prints when run on
sample_data/store_sales.csv. tests/test_graph.py runs the script through the real
executor and fails if a quoted number no longer appears in the code output.
"""

from pathlib import Path

from data_analyst_agent.replay import ScriptedQuestion, ScriptStep

SAMPLE_DATA_PATH = Path(__file__).resolve().parent / "sample_data" / "store_sales.csv"

_OVERVIEW_CODE = """\
import pandas as pd

df = pd.read_csv("data/store_sales.csv", parse_dates=["date"])
print(f"rows: {len(df):,}, columns: {df.shape[1]}")
print("columns:", ", ".join(df.columns))
print(f"dates: {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")
print("missing values per column:")
print(df.isna().sum().to_string())
by_region = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
print("revenue by region:")
print(by_region.map("{:,.0f}".format).to_string())
print(f"total revenue: {df['revenue'].sum():,.0f}")
"""

_MONTHLY_CHART_CODE = """\
import pandas as pd
import plotly.express as px

df = pd.read_csv("data/store_sales.csv", parse_dates=["date"])
monthly = (
    df.groupby([pd.Grouper(key="date", freq="MS"), "region"])["revenue"].sum().reset_index()
)
fig = px.line(
    monthly, x="date", y="revenue", color="region", markers=True,
    title="Monthly revenue by region", labels={"date": "Month", "revenue": "Revenue"},
)
fig.write_html("outputs/monthly_revenue_by_region.html")

table = monthly.pivot(index="date", columns="region", values="revenue")
table.index = table.index.strftime("%Y-%m")
print("first and last month:")
print(table.iloc[[0, -1]].map("{:,.0f}".format).to_string())
print("highest month per region:")
print(table.idxmax().to_string())
"""

# The first attempt uses the wrong column name on purpose, to show the agent recovering.
_GROWTH_FIRST_TRY_CODE = """\
import pandas as pd

df = pd.read_csv("data/store_sales.csv", parse_dates=["date"])
yearly = df.pivot_table(
    index="region", columns=df["date"].dt.year, values="Revenue", aggfunc="sum"
)
print(yearly)
"""

_GROWTH_CODE = """\
import numpy as np
import pandas as pd

df = pd.read_csv("data/store_sales.csv", parse_dates=["date"])
yearly = df.pivot_table(
    index="region", columns=df["date"].dt.year, values="revenue", aggfunc="sum"
)
yearly["growth"] = (yearly[2025] / yearly[2024] - 1) * 100

monthly = df.groupby([pd.Grouper(key="date", freq="MS"), "region"])["revenue"].sum().unstack()
month_number = np.arange(len(monthly))
yearly["trend_per_month"] = pd.Series(
    {region: np.polyfit(month_number, monthly[region], 1)[0] for region in monthly}
)
yearly = yearly.sort_values("growth", ascending=False)
print(yearly.to_string(formatters={
    2024: "{:,.0f}".format,
    2025: "{:,.0f}".format,
    "growth": "{:+.1f}%".format,
    "trend_per_month": "{:+,.0f}".format,
}))
"""

DEMO_SCRIPT = (
    ScriptedQuestion(
        question="Give me an overview of the sales data.",
        steps=(
            ScriptStep(
                "I will load the file and print its size, date range, missing values and "
                "revenue by region.",
                _OVERVIEW_CODE,
            ),
            ScriptStep(
                "The file has 8,772 rows and 6 columns (date, region, category, units, "
                "unit_price, revenue) and covers 2024-01-01 to 2025-12-31. Units and revenue "
                "are missing in 12 rows; the other columns are complete.\n\n"
                "Total revenue is 22,343,784. By region: East 6,912,030, North 6,013,439, "
                "South 5,058,144 and West 4,360,171."
            ),
        ),
    ),
    ScriptedQuestion(
        question="Plot monthly revenue by region and describe the trend.",
        steps=(
            ScriptStep(
                "I will sum revenue by month and region, save an interactive line chart to "
                "outputs/ and print the first and last month for each region.",
                _MONTHLY_CHART_CODE,
            ),
            ScriptStep(
                "The chart is saved as monthly_revenue_by_region.html.\n\n"
                "Monthly revenue was higher in 2025-12 than in 2024-01 in every region: "
                "West rose from 167,261 to 258,420, South from 212,794 to 270,059, North "
                "from 279,685 to 305,076 and East from 329,487 to 334,372. West, South and "
                "North reached their highest month in 2025-12; East peaked in 2025-01."
            ),
        ),
    ),
    ScriptedQuestion(
        question="Which region is growing fastest?",
        steps=(
            ScriptStep(
                "I will compare total revenue in 2025 with 2024 for each region.",
                _GROWTH_FIRST_TRY_CODE,
            ),
            ScriptStep(
                "The column is named `revenue` in lower case, so `Revenue` raised a KeyError. "
                "I will fix the name and also fit a linear trend to monthly revenue per region.",
                _GROWTH_CODE,
            ),
            ScriptStep(
                "West is growing fastest. Its revenue rose from 1,919,423 in 2024 to "
                "2,440,748 in 2025, a change of +27.2%, and its linear trend adds +3,838 per "
                "month. South follows at +12.1% (+2,320 per month), then North at +6.5% "
                "(+1,635 per month) and East at +2.4% (+805 per month).\n\n"
                "East still has the highest revenue in both years: 3,414,406 in 2024 and "
                "3,497,624 in 2025."
            ),
        ),
    ),
)

DEMO_QUESTIONS = tuple(q.question for q in DEMO_SCRIPT)
