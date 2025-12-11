"""
Economic Health Index (EHI) with Yield Curve Spread (standardized)
Interactive Plotly Version, class-based

Now with unified date handling:
  - We take the union of all available dates across:
      * labor+income+IP block
      * GDP
      * yield curve
      * money & total financial assets
  - Reindex everything to that full monthly index
  - Forward-fill each series to the latest date available.
"""

from datetime import datetime
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------
# Config
# ----------------------------
START_DATE = "1900-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

UNRATE_SERIES     = "UNRATE"
EMPLOYMENT_SERIES = "PAYEMS"
INCOME_SERIES     = "W875RX1"   # Real Disposable Personal Income
GDP_SERIES        = "GDPC1"
IP_SERIES         = "INDPRO"    # Industrial Production Index

DGS2_SERIES       = "DGS2"
DGS10_SERIES      = "DGS10"

# Money & investment series
M2_SERIES         = "M2SL"       # M2 Money Stock (monthly)
TFA_SERIES        = "TFAABSHNO"  # Households & Nonprofits; Total Financial Assets, Level (quarterly)


class EconomicHealthIndexPipeline:
    """
    Class-based pipeline to construct and plot the Economic Health Index (EHI),
    yield curve spread, and separate Money & Investment indices.
    """

    def __init__(self, start_date: str = START_DATE, end_date: str = END_DATE):
        self.start_date = start_date
        self.end_date = end_date

    # ----------------------------
    # Static / helper methods
    # ----------------------------

    @staticmethod
    def fetch_monthly_block(start=START_DATE, end=END_DATE):
        """
        Fetch monthly:
          - UNRATE
          - PAYEMS
          - W875RX1 (real disposable income)
          - INDPRO  (industrial production)
        """
        series = [UNRATE_SERIES, EMPLOYMENT_SERIES, INCOME_SERIES, IP_SERIES]
        df = pdr.DataReader(series, "fred", start=start, end=end)
        return df

    @staticmethod
    def fetch_quarterly_gdp(start=START_DATE, end=END_DATE):
        """
        Fetch quarterly real GDP (GDPC1) from FRED.
        """
        gdp = pdr.DataReader(GDP_SERIES, "fred", start=start, end=end)
        return gdp

    @staticmethod
    def convert_gdp_to_monthly(gdp_q: pd.DataFrame) -> pd.DataFrame:
        """
        Quarterly GDP → monthly via forward fill.
        Align index to month-start to match monthly indicators.
        """
        gdp_m = gdp_q.resample("M").ffill()
        gdp_m.index = gdp_m.index.to_period("M").to_timestamp()
        return gdp_m

    @staticmethod
    def fetch_yield_curve(start=START_DATE, end=END_DATE):
        """
        Fetch daily 2-year and 10-year Treasury yields, convert to monthly,
        and compute 2s/10s spread.
        """
        series = [DGS2_SERIES, DGS10_SERIES]
        yc = pdr.DataReader(series, "fred", start=start, end=end)

        # Month-end, take last available observation
        yc_m = yc.resample("M").last()
        yc_m.index = yc_m.index.to_period("M").to_timestamp()

        yc_m["spread_2s10s"] = yc_m[DGS10_SERIES] - yc_m[DGS2_SERIES]
        return yc_m

    @staticmethod
    def fetch_money_and_assets(start=START_DATE, end=END_DATE) -> pd.DataFrame:
        """
        Fetch:
          - M2 Money Stock (M2SL) [monthly, SA]
          - Households & Nonprofit Organizations; Total Financial Assets, Level (TFAABSHNO) [quarterly]
        Convert TFA to monthly via forward-fill.
        """
        # M2 is monthly already
        m2 = pdr.DataReader(M2_SERIES, "fred", start=start, end=end)

        # TFA is quarterly, convert to monthly
        tfa_q = pdr.DataReader(TFA_SERIES, "fred", start=start, end=end)
        tfa_m = tfa_q.resample("M").ffill()
        tfa_m.index = tfa_m.index.to_period("M").to_timestamp()

        # Align on union of both, forward-fill
        combined_index = m2.index.union(tfa_m.index).sort_values()
        m2 = m2.reindex(combined_index).ffill()
        tfa_m = tfa_m.reindex(combined_index).ffill()

        money_assets = m2.join(tfa_m, how="outer")
        return money_assets

    @staticmethod
    def compute_economic_health_index(df_monthly: pd.DataFrame,
                                      df_gdp: pd.DataFrame) -> pd.DataFrame:
        """
        Build the time-series Economic Health Index (EHI) using:
          - Labor (UNRATE, PAYEMS)
          - Income (W875RX1)
          - Real Activity (INDPRO, GDPC1)

        Returns DataFrame with:
          u, emp_mom, income_mom, ip_mom, gdp_mom
          z_u, z_emp, z_inc, z_ip, z_gdp
          LII, EHI, EHI_smooth
        """
        df = df_monthly.join(df_gdp, how="inner")
        print(f"Rows after join (macro block): {len(df)}")

        # Assign fields
        df["u"]      = df[UNRATE_SERIES]
        df["N"]      = df[EMPLOYMENT_SERIES]
        df["income"] = df[INCOME_SERIES]
        df["ip"]     = df[IP_SERIES]
        df["GDP"]    = df[GDP_SERIES]

        # Monthly % changes (simple rates)
        df["emp_mom"]    = df["N"] / df["N"].shift(1) - 1.0
        df["income_mom"] = df["income"] / df["income"].shift(1) - 1.0
        df["ip_mom"]     = df["ip"] / df["ip"].shift(1) - 1.0
        df["gdp_mom"]    = df["GDP"] / df["GDP"].shift(1) - 1.0

        # Drop initial NaNs
        df = df.dropna(subset=["u", "emp_mom", "income_mom", "ip_mom", "gdp_mom"])
        print(f"Rows after dropping NaNs (macro block): {len(df)}")

        # -------- Continuous normalization (full history) --------

        # Unemployment — lower is better
        df["z_u"] = (df["u"].mean() - df["u"]) / df["u"].std()

        # Employment MoM — higher is better
        df["z_emp"] = (df["emp_mom"] - df["emp_mom"].mean()) / df["emp_mom"].std()

        # Real disposable income MoM — higher is better
        df["z_inc"] = (df["income_mom"] - df["income_mom"].mean()) / df["income_mom"].std()

        # Industrial production MoM — higher is better
        df["z_ip"] = (df["ip_mom"] - df["ip_mom"].mean()) / df["ip_mom"].std()

        # GDP MoM — higher is better
        df["z_gdp"] = (df["gdp_mom"] - df["gdp_mom"].mean()) / df["gdp_mom"].std()

        # ---------------- Composite indices ----------------

        # Labor + Income Index (LII)
        df["LII"] = (df["z_u"] + df["z_emp"] + df["z_inc"]) / 3.0

        # Economic Health Index (EHI):
        # equal-weight of LII, GDP, and Industrial Production
        df["EHI"] = (df["LII"] + df["z_gdp"] + df["z_ip"]) / 3.0

        # Smoothed EHI (6-month moving average)
        df["EHI_smooth"] = df["EHI"].rolling(6, min_periods=1).mean()

        return df

    @staticmethod
    def plot_ehi_with_z_spread_plotly(df: pd.DataFrame) -> None:
        """
        Interactive Plotly plot with subplots:
          Top:
            - EHI
            - Smoothed EHI
            - z_spread_2s10s (standardized 2s/10s spread)
          Bottom:
            - Smoothed z(M2 growth)
            - Smoothed z(TFA growth)

        All axes constrained to [-4, 4] in standardized units.
        """
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.6, 0.4],
            subplot_titles=(
                "Economic Health Index & 2s/10s Yield Curve (z-scores)",
                "Money & Investment Indices (Smoothed z-scores)"
            ),
        )

        # ---- Top panel: EHI + yield curve ----
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EHI"],
                mode="lines",
                name="Economic Health Index",
                line=dict(width=1),
                opacity=0.5,
                hovertemplate="Date: %{x|%Y-%m-%d}<br>EHI: %{y:.2f}<extra></extra>",
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EHI_smooth"],
                mode="lines",
                name="Smoothed EHI (6-mo)",
                line=dict(width=2),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Smoothed EHI: %{y:.2f}<extra></extra>",
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["z_spread_2s10s"],
                mode="lines",
                name="2s/10s Spread (z-score)",
                line=dict(width=1, dash="dot"),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>z-Spread: %{y:.2f}<extra></extra>",
            ),
            row=1, col=1
        )

        # ---- Bottom panel: smoothed money & investment ----
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["z_m2_smooth"],
                mode="lines",
                name="M2 Growth (smoothed z)",
                line=dict(width=1),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>M2 z-sm: %{y:.2f}<extra></extra>",
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["z_tfa_smooth"],
                mode="lines",
                name="Total Financial Assets Growth (smoothed z)",
                line=dict(width=1, dash="dash"),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>TFA z-sm: %{y:.2f}<extra></extra>",
            ),
            row=2, col=1
        )

        # Horizontal zero line in each subplot
        fig.add_hline(y=0, line_width=1, line_dash="dash", opacity=0.6, row=1, col=1)
        fig.add_hline(y=0, line_width=1, line_dash="dash", opacity=0.6, row=2, col=1)

        # Constrain y-axes to [-4, 4]
        fig.update_yaxes(range=[-4, 4], row=1, col=1, title_text="Z-score")
        fig.update_yaxes(range=[-4, 4], row=2, col=1, title_text="Z-score")

        fig.update_layout(
            title="Economic Health, Yield Curve & Money/Investment Conditions",
            xaxis_title="Date",
            hovermode="x unified",
            legend=dict(x=0.01, y=0.99),
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="date"
            ),
            template="plotly_white",
        )

        fig.show()

    # ----------------------------
    # Orchestrator
    # ----------------------------

    def process(self) -> pd.DataFrame:
        """
        Run the full pipeline:
          - Fetch all data blocks
          - Build unified index & forward-fill
          - Compute EHI
          - Compute money & investment growth z-scores + smoothing
          - Standardize yield curve spread
          - Plot everything via Plotly (with subplots)

        Returns:
            df (pd.DataFrame): final DataFrame with all constructed series.
        """
        print("Fetching monthly labor + income + industrial production...")
        monthly_block = self.fetch_monthly_block(
            start=self.start_date, end=self.end_date
        )

        print("Fetching GDP data...")
        gdp_q = self.fetch_quarterly_gdp(
            start=self.start_date, end=self.end_date
        )
        print("Converting GDP to monthly...")
        gdp_m = self.convert_gdp_to_monthly(gdp_q)

        print("Fetching 2y/10y yield curve and computing spread...")
        yc_m = self.fetch_yield_curve(
            start=self.start_date, end=self.end_date
        )

        print("Fetching money supply (M2) and total financial assets...")
        money_assets = self.fetch_money_and_assets(
            start=self.start_date, end=self.end_date
        )

        # Unified date handling
        combined_index = (
            monthly_block.index
            .union(gdp_m.index)
            .union(yc_m.index)
            .union(money_assets.index)
        )
        combined_index = combined_index.sort_values()

        monthly_block = monthly_block.reindex(combined_index).ffill()
        gdp_m        = gdp_m.reindex(combined_index).ffill()
        yc_m         = yc_m.reindex(combined_index).ffill()
        money_assets = money_assets.reindex(combined_index).ffill()

        print(f"Unified index length: {len(combined_index)} "
              f"(from {combined_index[0].date()} to {combined_index[-1].date()})")

        # Compute EHI on unified macro panel
        print("Computing Economic Health Index (EHI)...")
        df = self.compute_economic_health_index(monthly_block, gdp_m)

        # Join spread into EHI dataframe (indexes now already aligned)
        df = df.join(yc_m["spread_2s10s"], how="left")

        # Standardize spread over its available history (z-score)
        spread = df["spread_2s10s"]
        spread_mean = spread.mean()
        spread_std = spread.std()
        df["z_spread_2s10s"] = (spread - spread_mean) / spread_std

        # Join money & assets and build separate indices
        df = df.join(money_assets[[M2_SERIES, TFA_SERIES]], how="left")

        # Monthly % changes for M2 and Total Financial Assets
        df["m2_mom"]  = df[M2_SERIES] / df[M2_SERIES].shift(1) - 1.0
        df["tfa_mom"] = df[TFA_SERIES] / df[TFA_SERIES].shift(1) - 1.0

        # z-scores (growth-based, like other flows)
        df["z_m2"] = (df["m2_mom"] - df["m2_mom"].mean()) / df["m2_mom"].std()
        df["z_tfa"] = (df["tfa_mom"] - df["tfa_mom"].mean()) / df["tfa_mom"].std()

        # Smoothed (6-month) z-scores for plotting
        window = 6
        df["z_m2_smooth"]  = df["z_m2"].rolling(window, min_periods=1).mean()
        df["z_tfa_smooth"] = df["z_tfa"].rolling(window, min_periods=1).mean()

        print("Latest observations:")
        print(df[[
            "u", "emp_mom", "income_mom", "ip_mom", "gdp_mom",
            "LII", "EHI", "EHI_smooth",
            "spread_2s10s", "z_spread_2s10s",
            "m2_mom", "tfa_mom", "z_m2", "z_tfa",
            "z_m2_smooth", "z_tfa_smooth"
        ]].tail())

        print("Rendering interactive Plotly chart...")
        self.plot_ehi_with_z_spread_plotly(df)

        return df


# ----------------------------
# Main
# ----------------------------

def main():
    pipeline = EconomicHealthIndexPipeline()
    df = pipeline.process()
    # df is returned if you want to inspect or save it externally
    return df


if __name__ == "__main__":
    main()
