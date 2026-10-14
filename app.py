import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.data_processor import (
    detect_date_columns,
    detect_numeric_columns,
    suggest_target_columns
)

from utils.forecast_engine import (
    prepare_time_series,
    forecast_arima
)

from utils.model_comparison import (
    compare_models
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart Demand Forecasting",
    page_icon="📈",
    layout="wide"
)


# =========================================================
# THEME STATE
# =========================================================

# The sidebar toggle updates this session state; the next Streamlit rerun
# applies the selected theme to the page and Plotly charts.
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

DARK_MODE = bool(st.session_state.get("dark_mode", False))
PLOTLY_TEMPLATE = "plotly_dark" if DARK_MODE else "plotly_white"


# =========================================================
# PROFESSIONAL CSS
# =========================================================

st.markdown(
    """
    <style>
    .stApp { background:#f7f9fc; }
    .block-container { max-width:1400px; padding-top:1.5rem; padding-bottom:3rem; }
    .hero {
        background:linear-gradient(135deg,#0f172a,#1e3a8a,#2563eb);
        padding:2.3rem 2.5rem; border-radius:20px; color:white;
        margin-bottom:1.5rem; box-shadow:0 12px 30px rgba(15,23,42,.12);
    }
    .hero-badge {
        display:inline-block; padding:.35rem .75rem; border-radius:999px;
        background:rgba(255,255,255,.13); font-size:.72rem;
        font-weight:800; letter-spacing:.1em;
    }
    .hero h1 { margin:.7rem 0 .4rem; font-size:2.65rem; font-weight:800; }
    .hero h1 span { color:#93c5fd; }
    .hero p { margin:0; color:#dbeafe; font-size:1rem; max-width:850px; }
    .section-label {
        color:#2563eb; font-size:.72rem; font-weight:800;
        letter-spacing:.1em; text-transform:uppercase;
    }
    .section-title { color:#0f172a; font-size:1.65rem; font-weight:800; margin:.1rem 0 .25rem; }
    .section-subtitle { color:#64748b; margin-bottom:1rem; }
    div[data-testid="stMetric"] {
        background:white; border:1px solid #e2e8f0; border-radius:14px;
        padding:.9rem 1rem; box-shadow:0 4px 14px rgba(15,23,42,.04);
    }
    .insight {
        background:white; border:1px solid #e2e8f0; border-left:5px solid #2563eb;
        border-radius:12px; padding:1rem 1.1rem; box-shadow:0 4px 14px rgba(15,23,42,.04);
    }
    .insight h4 { margin:0 0 .35rem; color:#0f172a; }
    .insight p { margin:0; color:#475569; line-height:1.55; }
    .stButton > button { border-radius:10px; font-weight:700; min-height:2.55rem; }
    .footer {
        text-align:center; color:#94a3b8; font-size:.8rem;
        padding-top:2rem; margin-top:2rem; border-top:1px solid #e2e8f0;
    }

    /* SmartDemand premium dashboard refinements */
    .block-container { max-width: 1500px; padding-top: 1.15rem; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eef4ff 0%, #f8faff 100%);
        border-right: 1px solid #dbe5f5;
    }
    .hero {
        background: radial-gradient(circle at 88% 12%, rgba(96,165,250,.38), transparent 28%),
                    linear-gradient(125deg, #0b1225 0%, #172554 48%, #1d4ed8 100%);
        border: 1px solid rgba(147,197,253,.24);
        padding: 2.45rem 2.6rem; border-radius: 24px;
        box-shadow: 0 18px 45px rgba(30,64,175,.17);
    }
    .hero h1 { letter-spacing: -.035em; }
    .hero-badge { border: 1px solid rgba(255,255,255,.18); }
    div[data-testid="stMetric"] {
        border: 1px solid #dce6f4; border-radius: 16px;
        background: linear-gradient(180deg,#ffffff 0%,#f9fbff 100%);
        padding: 1rem 1.05rem; box-shadow: 0 7px 22px rgba(15,23,42,.045);
        transition: transform .18s ease, box-shadow .18s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px); box-shadow: 0 12px 28px rgba(37,99,235,.10);
    }
    div[data-testid="stMetricLabel"] { color: #64748b; font-weight: 650; }
    div[data-testid="stMetricValue"] { color: #0f172a; font-weight: 800; }
    .stButton > button, .stDownloadButton > button {
        border-radius: 11px; font-weight: 700; min-height: 2.65rem;
        border: 1px solid #cbdaf0; transition: all .15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: #2563eb; color: #1d4ed8; box-shadow: 0 5px 15px rgba(37,99,235,.10);
    }
    div[data-testid="stDataFrame"] { border: 1px solid #dce6f4; border-radius: 13px; overflow: hidden; }
    .overview-banner {
        padding: 1.25rem 1.4rem; border-radius: 18px; color: white;
        background: linear-gradient(115deg,#111827 0%,#1e40af 62%,#3b82f6 100%);
        margin: .3rem 0 1rem 0; box-shadow: 0 12px 26px rgba(30,64,175,.14);
    }
    .overview-banner .eyebrow { font-size:.72rem; font-weight:800; letter-spacing:.13em; opacity:.82; }
    .overview-banner .headline { font-size:1.45rem; font-weight:800; margin:.25rem 0; }
    .overview-banner .subline { color:#dbeafe; font-size:.92rem; }
    .mini-note {
        background:#f0f7ff; border:1px solid #dbeafe; border-radius:12px;
        padding:.8rem 1rem; color:#1e3a8a; margin:.5rem 0 1rem 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Dark-mode overrides are intentionally added after the existing styles so
# the current dashboard design and all forecasting logic remain unchanged.
if DARK_MODE:
    st.markdown(
        """
        <style>
        .stApp, [data-testid="stAppViewContainer"] { background: #0b1120 !important; color: #e5e7eb !important; }
        [data-testid="stHeader"] { background: rgba(11,17,32,.92) !important; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg,#111827 0%,#0f172a 100%) !important; border-right: 1px solid #263449 !important; }
        [data-testid="stSidebar"] * { color: #e5e7eb; }
        .section-label { color: #93c5fd !important; }
        .section-title, h1, h2, h3, h4, p, label, [data-testid="stMarkdownContainer"] { color: #e5e7eb; }
        .section-subtitle, [data-testid="stCaptionContainer"] { color: #aab7ca !important; }
        div[data-testid="stMetric"] { background: linear-gradient(180deg,#172033 0%,#111827 100%) !important; border-color: #334155 !important; box-shadow: 0 7px 22px rgba(0,0,0,.16) !important; }
        div[data-testid="stMetricLabel"] { color: #cbd5e1 !important; }
        div[data-testid="stMetricValue"] { color: #f8fafc !important; }
        .insight { background: #172033 !important; border-color: #334155 !important; border-left-color: #60a5fa !important; }
        .insight h4, .insight p { color: #e5e7eb !important; }
        .mini-note { background: #172554 !important; border-color: #1d4ed8 !important; color: #dbeafe !important; }
        div[data-testid="stDataFrame"], div[data-testid="stTable"] { border-color: #334155 !important; }
        [data-testid="stExpander"] { background: #111827 !important; border-color: #334155 !important; }
        [data-baseweb="input"] > div, [data-baseweb="select"] > div, [data-baseweb="textarea"] { background: #111827 !important; border-color: #475569 !important; }
        input, textarea { color: #f8fafc !important; }
        .stButton > button, .stDownloadButton > button { background: #172033 !important; color: #f8fafc !important; border-color: #475569 !important; }
        .stButton > button:hover, .stDownloadButton > button:hover { border-color: #60a5fa !important; color: #bfdbfe !important; }
        hr { border-color: #334155 !important; }
        .footer { color: #94a3b8 !important; border-top-color: #334155 !important; }
        </style>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# SESSION STATE
# =========================================================

if "forecast_df" not in st.session_state:
    st.session_state.forecast_df = None

if "model_results" not in st.session_state:
    st.session_state.model_results = None

if "best_model" not in st.session_state:
    st.session_state.best_model = None

if "best_mape" not in st.session_state:
    st.session_state.best_mape = None

if "ai_recommendations" not in st.session_state:
    st.session_state.ai_recommendations = None


# =========================================================
# TITLE
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">AI-POWERED DEMAND ANALYTICS</div>
        <h1>Smart Demand <span>Forecasting</span></h1>
        <p>
            Transform historical business data into actionable demand forecasts
            using time-series analysis, ARIMA forecasting and model evaluation.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


with st.sidebar:
    st.markdown("## 📈 SmartDemand")
    st.caption("Demand Forecasting & Analytics")
    st.toggle("🌙 Dark mode", key="dark_mode", help="Switch between light and dark dashboard themes.")
    st.caption("Theme changes apply automatically on rerun.")
    st.markdown("---")
    st.markdown("### 📌 Workflow")
    st.markdown(
        "01  Upload CSV\n\n"
        "02  Select date & target\n\n"
        "03  Explore historical demand\n\n"
        "04  Compare models\n\n"
        "05  Generate forecast\n\n"
        "06  Review business insights\n\n"
        "07  AI recommendations\n\n"
        "08  Executive dashboard"
    )
    st.markdown("---")
    st.markdown("### ℹ️ Supported Data")
    st.caption(
        "Upload a CSV containing a date column and a numeric demand, "
        "sales or quantity column."
    )

# =========================================================
# FILE UPLOAD
# =========================================================

st.markdown('<div class="section-label">DATA INPUT</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Upload Your Business Dataset</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Start by uploading historical data in CSV format.</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)


# =========================================================
# PROCESS DATASET
# =========================================================

if uploaded_file is not None:

    try:

        # =================================================
        # LOAD DATA
        # =================================================

        df = pd.read_csv(uploaded_file)

        # If no real date column exists, only continue when the user confirms
        # that each row represents the next chronological time period.
        date_columns_detected = detect_date_columns(df)
        if not date_columns_detected:
            st.warning(
                "No actual date column was detected. Date-free forecasting is "
                "only appropriate when each row represents a time period and "
                "the rows are already in chronological order."
            )
            row_order = st.radio(
                "How are the sales records arranged?",
                [
                    "Rows are in chronological order",
                    "Rows are not in chronological order"
                ],
                key="date_free_row_order"
            )

            if row_order == "Rows are in chronological order":
                synthetic_date_col = "SmartDemand_Period"
                df[synthetic_date_col] = pd.date_range(
                    start="2000-01-01", periods=len(df), freq="D"
                )
                st.info(
                    "Date-free mode enabled. Synthetic dates preserve row order "
                    "but are placeholders, not actual calendar dates."
                )
            else:
                st.error(
                    "Time-series forecasting is unavailable because the records "
                    "are not confirmed to be chronological. Sort the records by "
                    "time before forecasting."
                )
                st.stop()

        st.success("✅ Dataset uploaded successfully!")


        # =================================================
        # DATASET PREVIEW
        # =================================================

        st.subheader("👀 Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True,
            hide_index=True
        )



        # =================================================
        # SMART DATA PROFILER
        # =================================================

        st.markdown('<div class="section-label">DATA INTELLIGENCE</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧠 Smart Data Profiler</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Automatically checks dataset quality and forecasting readiness before modeling.</div>',
            unsafe_allow_html=True
        )

        total_rows = int(df.shape[0])
        total_columns = int(df.shape[1])
        total_cells = max(total_rows * total_columns, 1)
        missing_cells = int(df.isnull().sum().sum())
        duplicate_rows = int(df.duplicated().sum())
        missing_pct = (missing_cells / total_cells) * 100
        duplicate_pct = (duplicate_rows / max(total_rows, 1)) * 100

        # Transparent quality score:
        # missing values and duplicates reduce the score.
        quality_score = 100.0
        quality_score -= min(missing_pct * 2.0, 30.0)
        quality_score -= min(duplicate_pct * 1.5, 15.0)
        quality_score = max(0.0, min(100.0, quality_score))

        if quality_score >= 85:
            quality_status = "Excellent ✅"
        elif quality_score >= 65:
            quality_status = "Needs Attention ⚠️"
        else:
            quality_status = "Poor ❌"

        numeric_count = len(detect_numeric_columns(df))
        profile_date_columns = detect_date_columns(df)
        if (
            "SmartDemand_Period" in df.columns
            and "SmartDemand_Period" not in profile_date_columns
        ):
            profile_date_columns.insert(0, "SmartDemand_Period")
        date_count = len(profile_date_columns)
        categorical_count = sum(
            1 for col in df.columns
            if str(df[col].dtype) in ("object", "category", "string")
        )

        p1, p2, p3, p4, p5 = st.columns(5)

        with p1:
            st.metric("Quality Score", f"{quality_score:.0f}%")

        with p2:
            st.metric("Rows", f"{total_rows:,}")

        with p3:
            st.metric("Missing Cells", f"{missing_cells:,}")

        with p4:
            st.metric("Duplicate Rows", f"{duplicate_rows:,}")

        with p5:
            st.metric("Data Types", f"{numeric_count} numeric")

        st.success(
            f"**Dataset Quality: {quality_status}**  •  "
            f"{date_count} date-like column(s)  •  "
            f"{numeric_count} numeric column(s)  •  "
            f"{categorical_count} categorical/text column(s)"
        )

        # Column-level profile
        with st.expander("📋 View detailed column profile", expanded=False):
            profile_rows = []

            for col in df.columns:
                series_col = df[col]
                profile_rows.append({
                    "Column": str(col),
                    "Data Type": str(series_col.dtype),
                    "Missing": int(series_col.isna().sum()),
                    "Missing %": round(float(series_col.isna().mean() * 100), 2),
                    "Unique Values": int(series_col.nunique(dropna=True)),
                    "Non-Null": int(series_col.notna().sum())
                })

            profile_df = pd.DataFrame(profile_rows)

            st.dataframe(
                profile_df,
                use_container_width=True,
                hide_index=True
            )

        # Automatic issue detection
        issues = []

        if missing_cells > 0:
            issues.append(
                f"⚠️ **{missing_cells:,} missing cell(s)** detected "
                f"({missing_pct:.2f}% of all cells)."
            )

        if duplicate_rows > 0:
            issues.append(
                f"⚠️ **{duplicate_rows:,} duplicate row(s)** detected "
                f"({duplicate_pct:.2f}% of records)."
            )

        if date_count == 0:
            issues.append("❌ No date-like column was detected.")
        if numeric_count == 0:
            issues.append("❌ No numeric column was detected.")

        if issues:
            with st.expander("⚠️ Data quality checks", expanded=True):
                for issue in issues:
                    st.markdown(issue)
        else:
            st.info("✅ No major data-quality issues detected.")

        # Forecast readiness
        readiness_ok = (
            total_rows >= 20 and
            date_count > 0 and
            numeric_count > 0
        )

        st.markdown("### 🎯 Forecast Readiness")

        if readiness_ok:
            st.success(
                f"🚀 **Forecast Ready** — The dataset contains {total_rows:,} records, "
                f"a date-like column and at least one numeric variable."
            )
        elif total_rows < 20:
            st.warning(
                f"⚠️ **Limited Forecast Readiness** — only {total_rows:,} records found. "
                "At least 20 historical observations are recommended."
            )
        else:
            st.warning(
                "⚠️ **Not Forecast Ready Yet** — the dataset needs a usable "
                "date column and numeric target column."
            )




        # =================================================
        # AUTOMATIC EDA & SEASONALITY DETECTION
        # =================================================

        st.markdown('<div class="section-label">AUTOMATED EXPLORATION</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Automatic EDA & Seasonality Detection</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">SmartDemand automatically summarizes demand behavior and checks for recurring patterns.</div>',
            unsafe_allow_html=True
        )

        eda_date_cols = detect_date_columns(df)
        if (
            "SmartDemand_Period" in df.columns
            and "SmartDemand_Period" not in eda_date_cols
        ):
            eda_date_cols.insert(0, "SmartDemand_Period")
        eda_numeric_cols = detect_numeric_columns(df)

        if eda_date_cols and eda_numeric_cols:
            eda_date_col = eda_date_cols[0]

            # Prefer the selected target if it already exists; otherwise use the first numeric column.
            eda_target_col = eda_numeric_cols[0]

            eda_df = df[[eda_date_col, eda_target_col]].copy()
            eda_df[eda_date_col] = pd.to_datetime(eda_df[eda_date_col], errors="coerce")
            eda_df[eda_target_col] = pd.to_numeric(eda_df[eda_target_col], errors="coerce")
            eda_df = eda_df.dropna().sort_values(eda_date_col)

            if len(eda_df) >= 2:
                eda_values = eda_df[eda_target_col]
                eda_dates = eda_df[eda_date_col]

                eda_mean = float(eda_values.mean())
                eda_median = float(eda_values.median())
                eda_min = float(eda_values.min())
                eda_max = float(eda_values.max())
                eda_std = float(eda_values.std()) if len(eda_values) > 1 else 0.0

                e1, e2, e3, e4 = st.columns(4)
                with e1:
                    st.metric("Average Demand", f"{eda_mean:,.0f}")
                with e2:
                    st.metric("Median Demand", f"{eda_median:,.0f}")
                with e3:
                    st.metric("Peak Demand", f"{eda_max:,.0f}")
                with e4:
                    st.metric("Demand Volatility", f"{eda_std:,.0f}")

                # Trend classification using first vs last rolling-window averages.
                window = max(3, min(12, len(eda_values) // 5))
                early_avg = float(eda_values.head(window).mean())
                recent_avg = float(eda_values.tail(window).mean())

                if early_avg != 0:
                    trend_change = ((recent_avg - early_avg) / abs(early_avg)) * 100
                else:
                    trend_change = 0.0

                if trend_change > 5:
                    trend_label = "Increasing 📈"
                elif trend_change < -5:
                    trend_label = "Decreasing 📉"
                else:
                    trend_label = "Stable ➡️"

                # Infer data frequency from the median date gap.
                date_gaps = eda_dates.diff().dropna().dt.total_seconds() / 86400
                median_gap = float(date_gaps.median()) if not date_gaps.empty else np.nan

                if pd.notna(median_gap) and median_gap <= 1.5:
                    frequency_label = "Daily"
                    candidate_periods = [7, 30]
                elif pd.notna(median_gap) and median_gap <= 8:
                    frequency_label = "Weekly"
                    candidate_periods = [4, 13, 26, 52]
                elif pd.notna(median_gap) and median_gap <= 35:
                    frequency_label = "Monthly"
                    candidate_periods = [3, 6, 12]
                elif pd.notna(median_gap) and median_gap <= 100:
                    frequency_label = "Quarterly"
                    candidate_periods = [4]
                else:
                    frequency_label = "Irregular"
                    candidate_periods = []

                # Estimate seasonality strength from lag correlations.
                seasonal_results = []
                for period in candidate_periods:
                    if len(eda_values) >= period * 2:
                        lag_corr = eda_values.autocorr(lag=period)
                        if pd.notna(lag_corr):
                            seasonal_results.append((period, float(lag_corr)))

                if seasonal_results:
                    best_period, best_corr = max(seasonal_results, key=lambda x: abs(x[1]))
                    seasonality_strength = abs(best_corr)

                    if seasonality_strength >= 0.70:
                        seasonality_label = "Strong 🔥"
                    elif seasonality_strength >= 0.40:
                        seasonality_label = "Moderate 🟡"
                    else:
                        seasonality_label = "Weak / Not clear ⚪"
                else:
                    best_period = None
                    best_corr = None
                    seasonality_strength = 0.0
                    seasonality_label = "Not enough data"

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.info(f"**Trend:** {trend_label}\n\nChange: **{trend_change:+.2f}%**")
                with c2:
                    st.info(f"**Frequency:** {frequency_label}\n\nMedian gap: **{median_gap:.1f} days**" if pd.notna(median_gap) else f"**Frequency:** {frequency_label}")
                with c3:
                    if best_period is not None:
                        st.info(
                            f"**Seasonality:** {seasonality_label}\n\n"
                            f"Best cycle: **{best_period} periods**"
                        )
                    else:
                        st.info(f"**Seasonality:** {seasonality_label}")

                # Interactive historical demand chart.
                fig_eda = go.Figure()
                fig_eda.add_trace(
                    go.Scatter(
                        x=eda_dates,
                        y=eda_values,
                        mode="lines+markers",
                        name="Demand"
                    )
                )

                if len(eda_values) >= window:
                    rolling = eda_values.rolling(window).mean()
                    fig_eda.add_trace(
                        go.Scatter(
                            x=eda_dates,
                            y=rolling,
                            mode="lines",
                            name=f"{window}-period moving average"
                        )
                    )

                fig_eda.update_layout(
                    title=f"Historical Demand — {eda_target_col}",
                    xaxis_title=eda_date_col,
                    yaxis_title=eda_target_col,
                    hovermode="x unified",
                    height=430,
                    margin=dict(l=20, r=20, t=55, b=20)
                )

                st.plotly_chart(fig_eda, use_container_width=True)

                # Demand distribution.
                with st.expander("📈 View demand distribution & automatic findings", expanded=False):
                    fig_dist = go.Figure()
                    fig_dist.add_trace(
                        go.Histogram(
                            x=eda_values,
                            nbinsx=20,
                            name="Demand distribution"
                        )
                    )
                    fig_dist.update_layout(
                        title="Demand Distribution",
                        xaxis_title=eda_target_col,
                        yaxis_title="Frequency",
                        height=350,
                        margin=dict(l=20, r=20, t=55, b=20)
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

                    findings = [
                        f"• Demand ranges from **{eda_min:,.0f}** to **{eda_max:,.0f}**.",
                        f"• Average demand is **{eda_mean:,.0f}**, while median demand is **{eda_median:,.0f}**.",
                        f"• The detected overall trend is **{trend_label}**.",
                        f"• Data frequency is estimated as **{frequency_label}**."
                    ]

                    if best_period is not None:
                        findings.append(
                            f"• The strongest tested recurring cycle is **{best_period} periods** "
                            f"with lag correlation **{best_corr:.2f}**."
                        )
                    else:
                        findings.append("• No reliable recurring seasonal cycle could be established from the available observations.")

                    for finding in findings:
                        st.markdown(finding)

            else:
                st.warning("⚠️ Automatic EDA needs at least two valid date-demand observations.")
        else:
            st.warning(
                "⚠️ Automatic EDA is waiting for a detectable date column and numeric demand column."
            )



        # =================================================
        # DEMAND ANOMALY DETECTION
        # =================================================

        st.markdown('<div class="section-label">DEMAND RISK INTELLIGENCE</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🚨 Demand Anomaly Detection</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">SmartDemand identifies unusual demand spikes and drops against a rolling statistical baseline.</div>',
            unsafe_allow_html=True
        )

        anomaly_date_cols = detect_date_columns(df)
        if (
            "SmartDemand_Period" in df.columns
            and "SmartDemand_Period" not in anomaly_date_cols
        ):
            anomaly_date_cols.insert(0, "SmartDemand_Period")
        anomaly_numeric_cols = detect_numeric_columns(df)

        if anomaly_date_cols and anomaly_numeric_cols:
            anomaly_date_col = anomaly_date_cols[0]
            anomaly_target_col = anomaly_numeric_cols[0]

            anomaly_df = df[[anomaly_date_col, anomaly_target_col]].copy()
            anomaly_df[anomaly_date_col] = pd.to_datetime(anomaly_df[anomaly_date_col], errors="coerce")
            anomaly_df[anomaly_target_col] = pd.to_numeric(anomaly_df[anomaly_target_col], errors="coerce")
            anomaly_df = anomaly_df.dropna().sort_values(anomaly_date_col).reset_index(drop=True)

            if len(anomaly_df) >= 10:
                values = anomaly_df[anomaly_target_col].astype(float)

                window = max(7, min(13, len(values) // 5))
                if window % 2 == 0:
                    window += 1

                baseline = values.rolling(
                    window=window,
                    center=True,
                    min_periods=max(3, window // 2)
                ).median()

                global_median = float(values.median())
                global_mad = float((values - global_median).abs().median())
                baseline = baseline.fillna(global_median)

                mad = (values - baseline).abs().rolling(
                    window=window,
                    center=True,
                    min_periods=max(3, window // 2)
                ).median()

                mad = mad.replace(0, np.nan).fillna(max(global_mad, 1e-9))
                robust_z = 0.6745 * (values - baseline) / mad

                high_mask = robust_z >= 3
                low_mask = robust_z <= -3
                anomaly_mask = high_mask | low_mask

                anomaly_df["Baseline"] = baseline
                anomaly_df["Anomaly Score"] = robust_z.abs()
                anomaly_df["Status"] = "Normal"
                anomaly_df.loc[high_mask, "Status"] = "High Spike"
                anomaly_df.loc[low_mask, "Status"] = "Low Drop"

                anomalies = anomaly_df[anomaly_mask].copy()

                a1, a2, a3, a4 = st.columns(4)
                with a1:
                    st.metric("Anomalies Detected", f"{len(anomalies):,}")
                with a2:
                    st.metric("High Spikes", f"{int(high_mask.sum()):,}")
                with a3:
                    st.metric("Low Drops", f"{int(low_mask.sum()):,}")
                with a4:
                    st.metric(
                        "Anomaly Rate",
                        f"{len(anomalies) / len(anomaly_df) * 100:.1f}%"
                    )

                if len(anomalies) == 0:
                    st.success(
                        "✅ **No significant demand anomalies detected.** "
                        "The historical demand pattern appears relatively consistent."
                    )
                else:
                    st.warning(
                        f"⚠️ **{len(anomalies)} unusual demand period(s) detected.** "
                        "These observations should be investigated before making inventory decisions."
                    )

                    display_anomalies = anomalies[
                        [anomaly_date_col, anomaly_target_col, "Baseline", "Anomaly Score", "Status"]
                    ].copy()

                    display_anomalies["Variance vs Baseline %"] = (
                        (
                            display_anomalies[anomaly_target_col]
                            - display_anomalies["Baseline"]
                        )
                        / display_anomalies["Baseline"].replace(0, np.nan)
                        * 100
                    ).round(2)

                    display_anomalies["Baseline"] = display_anomalies["Baseline"].round(2)
                    display_anomalies["Anomaly Score"] = display_anomalies["Anomaly Score"].round(2)

                    st.dataframe(
                        display_anomalies.sort_values("Anomaly Score", ascending=False),
                        use_container_width=True,
                        hide_index=True
                    )

                fig_anomaly = go.Figure()
                fig_anomaly.add_trace(
                    go.Scatter(
                        x=anomaly_df[anomaly_date_col],
                        y=anomaly_df[anomaly_target_col],
                        mode="lines",
                        name="Actual Demand"
                    )
                )
                fig_anomaly.add_trace(
                    go.Scatter(
                        x=anomaly_df[anomaly_date_col],
                        y=anomaly_df["Baseline"],
                        mode="lines",
                        name="Normal Baseline"
                    )
                )

                if high_mask.any():
                    high_points = anomaly_df[high_mask]
                    fig_anomaly.add_trace(
                        go.Scatter(
                            x=high_points[anomaly_date_col],
                            y=high_points[anomaly_target_col],
                            mode="markers",
                            name="High Spike",
                            marker=dict(size=11, symbol="triangle-up")
                        )
                    )

                if low_mask.any():
                    low_points = anomaly_df[low_mask]
                    fig_anomaly.add_trace(
                        go.Scatter(
                            x=low_points[anomaly_date_col],
                            y=low_points[anomaly_target_col],
                            mode="markers",
                            name="Low Drop",
                            marker=dict(size=11, symbol="triangle-down")
                        )
                    )

                fig_anomaly.update_layout(
                    title="Demand Anomaly Map",
                    xaxis_title=anomaly_date_col,
                    yaxis_title=anomaly_target_col,
                    hovermode="x unified",
                    height=430,
                    margin=dict(l=20, r=20, t=55, b=20)
                )

                st.plotly_chart(fig_anomaly, use_container_width=True)

                if len(anomalies) > 0:
                    strongest = anomalies.sort_values("Anomaly Score", ascending=False).iloc[0]
                    strongest_date = pd.to_datetime(strongest[anomaly_date_col]).strftime("%d %b %Y")
                    strongest_value = float(strongest[anomaly_target_col])
                    strongest_baseline = float(strongest["Baseline"])

                    if strongest["Status"] == "High Spike":
                        st.info(
                            f"🔺 **Priority Business Alert:** On **{strongest_date}**, "
                            f"demand reached **{strongest_value:,.0f}**, compared with "
                            f"a baseline of **{strongest_baseline:,.0f}**. "
                            "Investigate promotions, holidays, stock availability, or unusual customer demand."
                        )
                    else:
                        st.info(
                            f"🔻 **Priority Business Alert:** On **{strongest_date}**, "
                            f"demand fell to **{strongest_value:,.0f}**, compared with "
                            f"a baseline of **{strongest_baseline:,.0f}**. "
                            "Investigate weak demand, stock-outs, pricing, or operational issues."
                        )

                with st.expander("💡 How SmartDemand detects anomalies"):
                    st.write(
                        "Each observation is compared with a rolling median baseline. "
                        "A robust z-score is calculated using the median absolute deviation. "
                        "Values around ±3 or beyond are flagged as unusual."
                    )
            else:
                st.info(
                    "ℹ️ At least 10 valid historical observations are recommended for reliable anomaly detection."
                )
        else:
            st.warning(
                "⚠️ Anomaly Detection requires one detectable date column and at least one numeric demand column."
            )


        # =================================================
        # DATASET INFORMATION
        # =================================================

        st.subheader("📊 Dataset Information")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Rows",
                f"{df.shape[0]:,}"
            )

        with col2:
            st.metric(
                "Columns",
                df.shape[1]
            )

        with col3:
            st.metric(
                "Missing Values",
                f"{int(df.isnull().sum().sum()):,}"
            )

        with col4:
            st.metric(
                "Duplicate Rows",
                f"{int(df.duplicated().sum()):,}"
            )


        # =================================================
        # DETECT DATE COLUMN
        # =================================================

        date_columns = detect_date_columns(df)
        if (
            "SmartDemand_Period" in df.columns
            and "SmartDemand_Period" not in date_columns
        ):
            date_columns.insert(0, "SmartDemand_Period")

        st.subheader("📅 Date Column")

        if len(date_columns) > 0:

            selected_date = st.selectbox(
                "Select the date column",
                date_columns
            )

            st.success(
                f"Date column selected: **{selected_date}**"
            )

        else:

            st.error(
                "❌ No suitable date column detected."
            )

            st.stop()


        # =================================================
        # DETECT TARGET COLUMN
        # =================================================

        target_columns = suggest_target_columns(df)

        st.subheader("🎯 Forecast Target")

        if len(target_columns) > 0:

            selected_target = st.selectbox(
                "What do you want to forecast?",
                target_columns
            )

            st.success(
                f"Forecast target selected: **{selected_target}**"
            )

        else:

            st.error(
                "❌ No suitable numeric target detected."
            )

            st.stop()


        # =================================================
        # TARGET STATISTICS
        # =================================================

        st.subheader("📊 Target Statistics")

        target = pd.to_numeric(
            df[selected_target],
            errors="coerce"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Average",
                f"{target.mean():,.2f}"
            )

        with col2:
            st.metric(
                "Minimum",
                f"{target.min():,.2f}"
            )

        with col3:
            st.metric(
                "Maximum",
                f"{target.max():,.2f}"
            )

        with col4:
            st.metric(
                "Observations",
                f"{target.notna().sum():,}"
            )


        # =================================================
        # PREPARE TIME SERIES
        # =================================================

        time_series = prepare_time_series(
            df,
            selected_date,
            selected_target
        )


        # =========================================================
        # EXECUTIVE OVERVIEW — TOP-OF-PAGE DASHBOARD
        # =========================================================
        overview_series = time_series.copy()
        overview_series[selected_date] = pd.to_datetime(
            overview_series[selected_date], errors="coerce"
        )
        overview_series[selected_target] = pd.to_numeric(
            overview_series[selected_target], errors="coerce"
        )
        overview_series = (
            overview_series.dropna(subset=[selected_date, selected_target])
            .sort_values(selected_date)
        )

        st.markdown(
            """
            <div class="overview-banner">
                <div class="eyebrow">SMARTDEMAND • EXECUTIVE OVERVIEW</div>
                <div class="headline">Your demand, at a glance</div>
                <div class="subline">A live snapshot of the selected dataset before forecasting and business decisions.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        overview_values = overview_series[selected_target]
        overview_n = len(overview_values)
        overview_avg = float(overview_values.mean()) if overview_n else 0.0
        overview_latest = float(overview_values.iloc[-1]) if overview_n else 0.0
        overview_peak = float(overview_values.max()) if overview_n else 0.0
        overview_missing = int(df[[selected_date, selected_target]].isna().sum().sum())
        overview_recent_n = min(6, overview_n)
        overview_recent_avg = (
            float(overview_values.tail(overview_recent_n).mean())
            if overview_recent_n else 0.0
        )
        overview_previous_avg = (
            float(overview_values.iloc[-2 * overview_recent_n:-overview_recent_n].mean())
            if overview_n > overview_recent_n else overview_recent_avg
        )
        overview_change = (
            (overview_recent_avg - overview_previous_avg) / abs(overview_previous_avg) * 100
            if overview_previous_avg else 0.0
        )

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Average Demand", f"{overview_avg:,.0f}")
        k2.metric("Latest Demand", f"{overview_latest:,.0f}")
        k3.metric("Peak Demand", f"{overview_peak:,.0f}")
        k4.metric("Recent Trend", f"{overview_change:+.1f}%", help="Latest up to 6 observations compared with the previous block of observations.")
        k5.metric("Usable Records", f"{overview_n:,}", help=f"{overview_missing:,} missing cells in the selected date/target columns.")

        if overview_n >= 2:
            trend_label = (
                "rising" if overview_change >= 5
                else "falling" if overview_change <= -5
                else "relatively stable"
            )
            st.markdown(
                f'<div class="mini-note">📌 <b>Quick read:</b> Recent demand is <b>{trend_label}</b> '
                f'({overview_change:+.1f}% versus the previous comparison window). This is a descriptive signal, not a forecast.</div>',
                unsafe_allow_html=True
            )

            chart_left, chart_right = st.columns([1.7, 1])
            with chart_left:
                overview_fig = go.Figure()
                overview_fig.add_trace(go.Scatter(
                    x=overview_series[selected_date],
                    y=overview_series[selected_target],
                    mode="lines",
                    name="Historical demand",
                    line=dict(width=2.8, color="#2563eb"),
                    fill="tozeroy",
                    fillcolor="rgba(37,99,235,0.08)",
                    hovertemplate="%{x|%d %b %Y}<br>Demand: %{y:,.2f}<extra></extra>"
                ))
                if overview_n >= 6:
                    overview_roll = overview_series[selected_target].rolling(
                        min(6, max(2, overview_n // 5)), min_periods=1
                    ).mean()
                    overview_fig.add_trace(go.Scatter(
                        x=overview_series[selected_date],
                        y=overview_roll,
                        mode="lines",
                        name="Moving average",
                        line=dict(width=2, dash="dot", color="#f59e0b"),
                        hovertemplate="Moving average: %{y:,.2f}<extra></extra>"
                    ))
                overview_fig.update_layout(
                    title="Demand Pulse",
                    template=PLOTLY_TEMPLATE,
                    height=330,
                    margin=dict(l=10, r=10, t=55, b=10),
                    hovermode="x unified",
                    legend=dict(orientation="h", y=1.12, x=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title=None,
                    yaxis_title=str(selected_target)
                )
                st.plotly_chart(overview_fig, use_container_width=True)

            with chart_right:
                distribution_fig = go.Figure()
                distribution_fig.add_trace(go.Histogram(
                    x=overview_values,
                    nbinsx=min(18, max(6, int(np.sqrt(overview_n)))),
                    marker=dict(color="#60a5fa", line=dict(color="#ffffff", width=1)),
                    name="Observations",
                    hovertemplate="Demand range: %{x}<br>Records: %{y}<extra></extra>"
                ))
                distribution_fig.update_layout(
                    title="Demand Distribution",
                    template=PLOTLY_TEMPLATE,
                    height=330,
                    margin=dict(l=10, r=10, t=55, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title=str(selected_target),
                    yaxis_title="Records",
                    showlegend=False
                )
                st.plotly_chart(distribution_fig, use_container_width=True)

            with st.expander("🔎 View executive snapshot details"):
                snapshot = pd.DataFrame([{
                    "Selected date column": str(selected_date),
                    "Selected target": str(selected_target),
                    "First observation": str(overview_series[selected_date].iloc[0].date()),
                    "Latest observation": str(overview_series[selected_date].iloc[-1].date()),
                    "Usable observations": overview_n,
                    "Average demand": overview_avg,
                    "Median demand": float(overview_values.median()),
                    "Minimum demand": float(overview_values.min()),
                    "Maximum demand": overview_peak,
                    "Standard deviation": float(overview_values.std()) if overview_n > 1 else 0.0,
                    "Recent change %": overview_change
                }])
                st.dataframe(snapshot, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download overview snapshot",
                    data=snapshot.to_csv(index=False),
                    file_name="smartdemand_overview_snapshot.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        st.markdown("---")

        if len(time_series) < 20:

            st.warning(
                "⚠️ At least 20 historical observations are recommended "
                "for reliable forecasting."
            )


        # =================================================
        # HISTORICAL TREND
        # =================================================

        st.subheader("📈 Historical Demand Trend")

        historical = time_series.copy()

        historical[selected_date] = pd.to_datetime(
            historical[selected_date],
            errors="coerce"
        )

        historical[selected_target] = pd.to_numeric(
            historical[selected_target],
            errors="coerce"
        )

        historical = historical.dropna().sort_values(
            selected_date
        )

        fig_hist = go.Figure()

        fig_hist.add_trace(
            go.Scatter(
                x=historical[selected_date],
                y=historical[selected_target],
                mode="lines",
                name="Historical Demand",
                line=dict(width=2)
            )
        )

        fig_hist.update_layout(
            title="Historical Demand Over Time",
            xaxis_title="Date",
            yaxis_title=selected_target,
            hovermode="x unified",
            template=PLOTLY_TEMPLATE,
            height=450
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True
        )


        # =================================================
        # MODEL COMPARISON
        # =================================================

        st.subheader("📊 Forecasting Model Comparison")

        st.write(
            "The system compares forecasting models using MAE, RMSE "
            "and MAPE. Lower error values indicate better performance."
        )

        if st.button(
            "🔍 Compare Forecasting Models"
        ):

            if len(time_series) < 20:

                st.error(
                    "❌ Not enough historical data for model comparison."
                )

            else:

                with st.spinner(
                    "Comparing ARIMA and Exponential Smoothing..."
                ):

                    try:

                        # IMPORTANT:
                        # compare_models expects the one-dimensional
                        # target series, not the complete dataframe.

                        series = time_series[
                            selected_target
                        ].dropna()

                        comparison = compare_models(series)

                        if len(comparison) == 2:
                            results, best_model = comparison
                            if isinstance(best_model, dict):
                                best_model_name = best_model.get("Model")
                                best_mape = best_model.get("MAPE")
                            else:
                                best_model_name = best_model
                                best_mape = None
                        elif len(comparison) == 3:
                            results, best_model, best_mape = comparison
                            if isinstance(best_model, dict):
                                best_model_name = best_model.get("Model")
                                if best_mape is None:
                                    best_mape = best_model.get("MAPE")
                            else:
                                best_model_name = best_model
                        else:
                            raise ValueError("Unexpected model comparison output.")

                        if len(results) > 0:
                            st.session_state.model_results = results.copy()
                            st.session_state.best_model = best_model_name
                            st.session_state.best_mape = best_mape
                            st.success("✅ Model comparison completed!")

                        else:

                            st.error(
                                "❌ Unable to compare forecasting models."
                            )

                    except Exception as model_error:

                        st.error(
                            f"❌ Model comparison failed: {model_error}"
                        )


        # =================================================
        # DISPLAY MODEL RESULTS
        # =================================================

        if st.session_state.model_results is not None:

            results = st.session_state.model_results.copy()

            st.markdown("### 📋 Model Performance")

            for column in ["MAE", "RMSE", "MAPE"]:

                if column in results.columns:

                    results[column] = pd.to_numeric(
                        results[column],
                        errors="coerce"
                    ).round(2)

            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True
            )


            if st.session_state.best_model is not None:

                st.markdown("### 🏆 Best Forecasting Model")

                best_model = st.session_state.best_model
                best_mape = st.session_state.best_mape

                st.success(
                    f"Best Model: **{best_model}**"
                )

                if "Model" in results.columns:

                    best_rows = results[
                        results["Model"].astype(str) ==
                        str(best_model)
                    ]

                    if not best_rows.empty:

                        best_row = best_rows.iloc[0]

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            if "MAE" in best_row:
                                st.metric(
                                    "MAE",
                                    f"{float(best_row['MAE']):,.2f}"
                                )

                        with col2:
                            if "RMSE" in best_row:
                                st.metric(
                                    "RMSE",
                                    f"{float(best_row['RMSE']):,.2f}"
                                )

                        with col3:
                            if "MAPE" in best_row:
                                st.metric(
                                    "MAPE",
                                    f"{float(best_row['MAPE']):,.2f}%"
                                )
                            elif best_mape is not None:
                                st.metric(
                                    "MAPE",
                                    f"{float(best_mape):,.2f}%"
                                )


        # =================================================
        # FUTURE FORECAST
        # =================================================

        st.subheader("🔮 Future Demand Forecast")

        col1, col2 = st.columns([2, 1])

        with col1:

            forecast_periods = st.number_input(
                "Number of future periods",
                min_value=1,
                max_value=52,
                value=12,
                step=1
            )

        with col2:

            st.metric(
                "Forecast Horizon",
                f"{forecast_periods} periods"
            )

        st.write(
            "Choose how many future periods you want to predict."
        )


        if st.button(
            "🚀 Generate Future Forecast"
        ):

            if len(time_series) < 20:

                st.error(
                    "❌ Not enough historical data for reliable forecasting."
                )

            else:

                with st.spinner(
                    "Training ARIMA model and generating forecast..."
                ):

                    try:

                        forecast_df = forecast_arima(
                            time_series,
                            selected_date,
                            selected_target,
                            periods=forecast_periods
                        )

                        forecast_df = forecast_df.copy()

                        forecast_df["Date"] = pd.to_datetime(
                            forecast_df["Date"],
                            errors="coerce"
                        )

                        forecast_df["Forecast"] = pd.to_numeric(
                            forecast_df["Forecast"],
                            errors="coerce"
                        )

                        forecast_df = forecast_df.dropna(
                            subset=["Date", "Forecast"]
                        )

                        st.session_state.forecast_df = forecast_df

                        st.success(
                            "✅ Forecast generated successfully!"
                        )

                    except Exception as forecast_error:

                        st.error(
                            f"❌ Forecast generation failed: {forecast_error}"
                        )


        # =================================================
        # FORECAST DASHBOARD
        # =================================================

        if st.session_state.forecast_df is not None:

            forecast_display = (
                st.session_state.forecast_df.copy()
            )

            st.subheader("📊 Forecast Dashboard")


            # =================================================
            # FORECAST KPI CALCULATIONS
            # =================================================

            total_forecast = forecast_display[
                "Forecast"
            ].sum()

            average_forecast = forecast_display[
                "Forecast"
            ].mean()

            highest_forecast = forecast_display[
                "Forecast"
            ].max()

            lowest_forecast = forecast_display[
                "Forecast"
            ].min()


            highest_date = forecast_display.loc[
                forecast_display["Forecast"].idxmax(),
                "Date"
            ]

            lowest_date = forecast_display.loc[
                forecast_display["Forecast"].idxmin(),
                "Date"
            ]


            # =================================================
            # TREND CALCULATION
            # =================================================

            first_forecast = forecast_display[
                "Forecast"
            ].iloc[0]

            last_forecast = forecast_display[
                "Forecast"
            ].iloc[-1]

            if first_forecast != 0:

                trend_percentage = (
                    (last_forecast - first_forecast)
                    / abs(first_forecast)
                ) * 100

            else:

                trend_percentage = 0


            if trend_percentage > 2:

                trend_label = "📈 Increasing"

            elif trend_percentage < -2:

                trend_label = "📉 Decreasing"

            else:

                trend_label = "➡️ Stable"


            # =================================================
            # KPI CARDS
            # =================================================

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "📅 Forecast Periods",
                    len(forecast_display)
                )

            with col2:

                st.metric(
                    "💰 Total Forecast",
                    f"{total_forecast:,.0f}"
                )

            with col3:

                st.metric(
                    "📊 Average Demand",
                    f"{average_forecast:,.0f}"
                )

            with col4:

                st.metric(
                    "📈 Peak Demand",
                    f"{highest_forecast:,.0f}"
                )


            # =================================================
            # HISTORICAL VS FUTURE FORECAST
            # =================================================

            st.subheader(
                "📈 Historical vs Future Demand"
            )

            fig_forecast = go.Figure()


            # Historical line

            fig_forecast.add_trace(
                go.Scatter(
                    x=historical[selected_date],
                    y=historical[selected_target],
                    mode="lines",
                    name="Historical Demand",
                    line=dict(width=2)
                )
            )


            # Forecast line

            fig_forecast.add_trace(
                go.Scatter(
                    x=forecast_display["Date"],
                    y=forecast_display["Forecast"],
                    mode="lines+markers",
                    name="Future Forecast",
                    line=dict(
                        width=3,
                        dash="dash"
                    ),
                    marker=dict(
                        size=7
                    )
                )
            )


            # Forecast starting point

            fig_forecast.add_trace(
                go.Scatter(
                    x=[
                        historical[selected_date].iloc[-1],
                        forecast_display["Date"].iloc[0]
                    ],
                    y=[
                        historical[selected_target].iloc[-1],
                        forecast_display["Forecast"].iloc[0]
                    ],
                    mode="lines",
                    name="Forecast Transition",
                    line=dict(
                        width=2,
                        dash="dot"
                    ),
                    showlegend=False
                )
            )


            fig_forecast.update_layout(
                title=(
                    f"{selected_target}: Historical Trend "
                    "and Future Forecast"
                ),
                xaxis_title="Date",
                yaxis_title=selected_target,
                hovermode="x unified",
                template=PLOTLY_TEMPLATE,
                height=550,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0
                )
            )


            st.plotly_chart(
                fig_forecast,
                use_container_width=True
            )


            # =================================================
            # FORECAST TABLE
            # =================================================

            st.subheader("📋 Future Predictions")

            display_table = forecast_display.copy()

            display_table["Date"] = (
                display_table["Date"]
                .dt.strftime("%d %b %Y")
            )

            display_table["Forecast"] = (
                display_table["Forecast"]
                .round(2)
            )

            st.dataframe(
                display_table,
                use_container_width=True,
                hide_index=True
            )


            # =================================================
            # PEAK & LOW DEMAND
            # =================================================

            col1, col2 = st.columns(2)

            with col1:

                st.info(
                    f"""
                    📈 **Highest Predicted Demand**

                    Date: **{highest_date.strftime('%d %B %Y')}**

                    Expected demand: **{highest_forecast:,.2f}**
                    """
                )

            with col2:

                st.warning(
                    f"""
                    📉 **Lowest Predicted Demand**

                    Date: **{lowest_date.strftime('%d %B %Y')}**

                    Expected demand: **{lowest_forecast:,.2f}**
                    """
                )


            # =================================================
            # BUSINESS INSIGHTS
            # =================================================

            st.subheader("💼 Business Insights")


            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Demand Trend",
                    trend_label
                )

            with col2:

                st.metric(
                    "Trend Change",
                    f"{trend_percentage:+.2f}%"
                )

            with col3:

                if st.session_state.best_model:

                    st.metric(
                        "Best Model",
                        st.session_state.best_model
                    )

                else:

                    st.metric(
                        "Best Model",
                        "ARIMA"
                    )


            # =================================================
            # MANAGEMENT SUMMARY
            # =================================================

            st.markdown(
                "### 📌 Management Summary"
            )

            if trend_percentage > 2:

                recommendation = (
                    "Forecasted demand is increasing. "
                    "The business should maintain sufficient inventory, "
                    "prepare resources for higher demand and monitor "
                    "stock levels to reduce the risk of shortages."
                )

            elif trend_percentage < -2:

                recommendation = (
                    "Forecasted demand is decreasing. "
                    "The business should avoid excessive inventory, "
                    "review purchasing quantities and monitor demand "
                    "before committing additional resources."
                )

            else:

                recommendation = (
                    "Forecasted demand is relatively stable. "
                    "The business can maintain inventory close to "
                    "expected demand while monitoring future changes."
                )


            st.info(
                f"""
                **Forecast Analysis**

                • Expected average demand:
                **{average_forecast:,.2f}**

                • Total expected demand:
                **{total_forecast:,.2f}**

                • Highest expected demand:
                **{highest_forecast:,.2f}**
                on **{highest_date.strftime('%d %B %Y')}**

                • Lowest expected demand:
                **{lowest_forecast:,.2f}**
                on **{lowest_date.strftime('%d %B %Y')}**

                • Overall demand trend:
                **{trend_label}**

                • Forecast change from first to last period:
                **{trend_percentage:+.2f}%**
                """
            )


            # =================================================
            # BUSINESS RECOMMENDATION
            # =================================================

            st.markdown(
                "### 🎯 Business Recommendation"
            )

            st.success(
                f"💼 {recommendation}"
            )


            # =================================================
            # DOWNLOAD FORECAST
            # =================================================

            csv = forecast_display.to_csv(
                index=False
            )

            st.download_button(
                label="⬇️ Download Future Forecast CSV",
                data=csv,
                file_name="future_demand_forecast.csv",
                mime="text/csv",
                use_container_width=True
            )



        # ============================================================
        # SMART INVENTORY RECOMMENDATION
        # ============================================================
        st.markdown("---")
        st.header("📦 Smart Inventory Recommendation")
        st.caption(
            "Convert the demand forecast into a practical inventory planning decision."
        )

        inventory_forecast = st.session_state.get("forecast_df")

        if inventory_forecast is None or inventory_forecast.empty:
            st.info(
                "💡 Generate the future forecast above to unlock the inventory recommendation."
            )
        else:
            try:
                forecast_values = pd.to_numeric(
                    inventory_forecast["Forecast"],
                    errors="coerce"
                ).dropna().to_numpy(dtype=float)

                forecast_values = forecast_values[np.isfinite(forecast_values)]

                if len(forecast_values) > 0:

                    avg_forecast = float(np.mean(forecast_values))
                    total_forecast = float(np.sum(forecast_values))

                    historical_values = pd.to_numeric(
                        time_series[selected_target],
                        errors="coerce"
                    ).dropna().to_numpy(dtype=float)

                    historical_values = historical_values[
                        np.isfinite(historical_values)
                    ]

                    recent_n = min(12, len(historical_values))

                    recent_avg = (
                        float(np.mean(historical_values[-recent_n:]))
                        if recent_n > 0
                        else 0.0
                    )

                    historical_std = (
                        float(np.std(historical_values, ddof=1))
                        if len(historical_values) > 1
                        else 0.0
                    )

                    # Simple planning assumption
                    safety_buffer = float(1.65 * historical_std)

                    recommended_stock = (
                        total_forecast + safety_buffer
                    )

                    demand_change_pct = (
                        ((avg_forecast - recent_avg) / recent_avg) * 100
                        if recent_avg > 0
                        else 0.0
                    )

                    if demand_change_pct >= 10:

                        inventory_status = "🟢 Increase Stock"

                        inventory_message = (
                            f"Forecast demand is {demand_change_pct:.1f}% "
                            "higher than recent demand. Increase inventory "
                            "planning to reduce potential stock-shortage risk."
                        )

                    elif demand_change_pct <= -10:

                        inventory_status = "🟡 Reduce / Control Stock"

                        inventory_message = (
                            f"Forecast demand is {abs(demand_change_pct):.1f}% "
                            "lower than recent demand. Control inventory "
                            "to reduce potential excess-stock risk."
                        )

                    else:

                        inventory_status = "🔵 Maintain Balanced Stock"

                        inventory_message = (
                            f"Forecast demand is {abs(demand_change_pct):.1f}% "
                            "different from recent demand. A balanced "
                            "inventory plan is appropriate."
                        )

                    c1, c2, c3, c4 = st.columns(4)

                    c1.metric(
                        "Average Forecast Demand",
                        f"{avg_forecast:,.0f}"
                    )

                    c2.metric(
                        "Forecast Horizon Demand",
                        f"{total_forecast:,.0f}"
                    )

                    c3.metric(
                        "Planning Safety Buffer",
                        f"{safety_buffer:,.0f}"
                    )

                    c4.metric(
                        "Recommended Horizon Stock",
                        f"{recommended_stock:,.0f}"
                    )

                    st.success(
                        f"**Inventory Decision: {inventory_status}**"
                    )

                    st.info(inventory_message)

                    inventory_breakdown = pd.DataFrame({
                        "Planning Metric": [
                            "Recent Average Demand",
                            "Average Forecast Demand",
                            "Forecast Horizon Demand",
                            "Planning Safety Buffer",
                            "Recommended Horizon Stock"
                        ],
                        "Value": [
                            recent_avg,
                            avg_forecast,
                            total_forecast,
                            safety_buffer,
                            recommended_stock
                        ]
                    })

                    st.subheader("📋 Inventory Planning Breakdown")

                    st.dataframe(
                        inventory_breakdown.style.format(
                            {"Value": "{:,.0f}"}
                        ),
                        use_container_width=True,
                        hide_index=True
                    )

                    with st.expander(
                        "ℹ️ How is the recommendation calculated?"
                    ):

                        st.write(
                            "Recommended horizon stock = total forecast "
                            "demand + planning safety buffer."
                        )

                        st.write(
                            "Safety buffer = 1.65 × historical demand "
                            "standard deviation. This is a simple planning "
                            "assumption, not a guaranteed service level."
                        )

                        st.write(
                            f"The forecast is compared with the latest "
                            f"{recent_n} historical observations to determine "
                            "the inventory action."
                        )

                else:

                    st.warning(
                        "No valid forecast values were available "
                        "for inventory planning."
                    )

            except Exception as inventory_error:

                st.warning(
                    f"Inventory recommendation unavailable: {inventory_error}"
                )

        # ============================================================
        # AI BUSINESS RECOMMENDATION ENGINE
        # ============================================================
        st.markdown("---")
        st.markdown('<div class="section-label">DECISION INTELLIGENCE</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🤖 AI Business Recommendation Engine</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Combines forecast, trend, anomalies and model signals into clear business actions.</div>', unsafe_allow_html=True)

        ai_forecast_frame = st.session_state.get("forecast_df")
        if ai_forecast_frame is None or ai_forecast_frame.empty:
            st.info("💡 Generate the future forecast above to unlock AI recommendations.")
        else:
            ai_forecast_frame = ai_forecast_frame.copy()
            ai_forecast_frame["Forecast"] = pd.to_numeric(ai_forecast_frame["Forecast"], errors="coerce")
            ai_forecast_frame = ai_forecast_frame.dropna(subset=["Forecast"])

            ai_recent_values = pd.to_numeric(time_series[selected_target], errors="coerce").dropna()
            ai_recent_n = min(12, len(ai_recent_values))
            ai_recent_avg = float(ai_recent_values.tail(ai_recent_n).mean()) if ai_recent_n else 0.0
            ai_forecast_avg = float(ai_forecast_frame["Forecast"].mean()) if not ai_forecast_frame.empty else 0.0
            ai_change_pct = ((ai_forecast_avg - ai_recent_avg) / abs(ai_recent_avg) * 100) if ai_recent_avg else 0.0

            recommendation_items = []
            if ai_change_pct >= 10:
                recommendation_items.append({"Priority":"High","Area":"Demand","Signal":f"Forecast is {ai_change_pct:.1f}% above recent demand.","Action":"Prepare additional inventory and procurement capacity."})
            elif ai_change_pct <= -10:
                recommendation_items.append({"Priority":"Medium","Area":"Demand","Signal":f"Forecast is {abs(ai_change_pct):.1f}% below recent demand.","Action":"Control purchasing and monitor excess-stock risk."})
            else:
                recommendation_items.append({"Priority":"Low","Area":"Demand","Signal":f"Forecast is {abs(ai_change_pct):.1f}% different from recent demand.","Action":"Maintain a balanced plan and continue monitoring."})

            if len(ai_recent_values) >= 10:
                ai_window = max(3, min(12, len(ai_recent_values) // 5))
                ai_early = float(ai_recent_values.head(ai_window).mean())
                ai_late = float(ai_recent_values.tail(ai_window).mean())
                ai_trend_pct = ((ai_late - ai_early) / abs(ai_early) * 100) if ai_early else 0.0
                if ai_trend_pct > 5:
                    recommendation_items.append({"Priority":"High","Area":"Trend","Signal":f"Historical demand increased by about {ai_trend_pct:.1f}%.","Action":"Monitor capacity and prepare for higher demand."})
                elif ai_trend_pct < -5:
                    recommendation_items.append({"Priority":"Medium","Area":"Trend","Signal":f"Historical demand decreased by about {abs(ai_trend_pct):.1f}%.","Action":"Review demand drivers and avoid unnecessary inventory buildup."})
                else:
                    recommendation_items.append({"Priority":"Low","Area":"Trend","Signal":"Historical demand is relatively stable.","Action":"Use the forecast as a planning reference and monitor changes."})

            ai_mape = st.session_state.get("best_mape")
            if ai_mape is not None:
                try:
                    ai_mape = float(ai_mape)
                    priority = "Low" if ai_mape <= 10 else "Medium" if ai_mape <= 20 else "High"
                    action = "Use as a planning baseline with routine monitoring." if ai_mape <= 10 else "Review performance periodically." if ai_mape <= 20 else "Treat cautiously and investigate model/data limitations."
                    recommendation_items.append({"Priority":priority,"Area":"Model Reliability","Signal":f"Selected model MAPE is {ai_mape:.2f}%.","Action":action})
                except (TypeError, ValueError):
                    pass

            high_count = sum(item["Priority"] == "High" for item in recommendation_items)
            medium_count = sum(item["Priority"] == "Medium" for item in recommendation_items)
            a1, a2, a3 = st.columns(3)
            a1.metric("High-Priority Actions", high_count)
            a2.metric("Monitoring Actions", medium_count)
            a3.metric("Signals Reviewed", len(recommendation_items))
            main_action = next((item for item in recommendation_items if item["Priority"] == "High"), recommendation_items[0])
            st.info(f"**Recommended Focus — {main_action['Area']}:** {main_action['Action']}")
            st.subheader("🧠 Explainable Recommendation Breakdown")
            st.dataframe(pd.DataFrame(recommendation_items), use_container_width=True, hide_index=True)
            with st.expander("ℹ️ How does the AI recommendation engine work?"):
                st.write("This is transparent, rule-based decision intelligence using forecast-versus-recent demand, historical trend and model MAPE. It provides planning suggestions, not guaranteed outcomes.")

        # ============================================================
        # EXECUTIVE MANAGEMENT DASHBOARD
        # ============================================================
        st.markdown("---")
        st.markdown('<div class="section-label">EXECUTIVE OVERVIEW</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Executive Management Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">A single-screen summary of forecast, model, demand risk and inventory planning.</div>', unsafe_allow_html=True)

        exec_forecast = st.session_state.get("forecast_df")
        if exec_forecast is None or exec_forecast.empty:
            st.info("💡 Generate the future forecast above to unlock the Executive Dashboard.")
        else:
            exec_forecast = exec_forecast.copy()
            exec_forecast["Forecast"] = pd.to_numeric(exec_forecast["Forecast"], errors="coerce")
            exec_forecast = exec_forecast.dropna(subset=["Forecast"])
            if not exec_forecast.empty:
                exec_avg = float(exec_forecast["Forecast"].mean())
                exec_total = float(exec_forecast["Forecast"].sum())
                exec_history = pd.to_numeric(time_series[selected_target], errors="coerce").dropna()
                exec_recent = float(exec_history.tail(min(12, len(exec_history))).mean()) if len(exec_history) else 0.0
                exec_change = ((exec_avg - exec_recent) / abs(exec_recent) * 100) if exec_recent else 0.0
                exec_model = st.session_state.get("best_model") or "Not compared"
                exec_mape = st.session_state.get("best_mape")
                exec_std = float(exec_history.std()) if len(exec_history) > 1 else 0.0
                exec_stock = exec_total + max(0.0, 1.65 * exec_std)
                exec_action = "Increase Stock" if exec_change >= 10 else "Control Stock" if exec_change <= -10 else "Maintain Balanced Stock"

                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Average Forecast", f"{exec_avg:,.0f}")
                b2.metric("Forecast Change", f"{exec_change:+.1f}%")
                b3.metric("Forecast Horizon", f"{len(exec_forecast)} periods")
                b4.metric("Recommended Stock", f"{exec_stock:,.0f}")
                b5, b6, b7, b8 = st.columns(4)
                b5.metric("Best Model", str(exec_model))
                b6.metric("MAPE", f"{float(exec_mape):.2f}%" if exec_mape is not None else "Not available")
                b7.metric("Total Forecast Demand", f"{exec_total:,.0f}")
                b8.metric("Recent Average Demand", f"{exec_recent:,.0f}")

                st.subheader("🧾 Executive Business Summary")
                st.info(f"**Demand outlook:** {'Increasing' if exec_change >= 5 else 'Decreasing' if exec_change <= -5 else 'Relatively stable'}\n\n**Inventory action:** {exec_action}\n\n**Forecast horizon demand:** {exec_total:,.0f}\n\n**Selected model:** {exec_model}\n\n**Planning note:** Forecasts are estimates; review model error and business context before acting.")
                st.subheader("💡 Key Takeaways")
                for item in [
                    f"Average forecast demand is {exec_avg:,.0f} per period.",
                    f"Forecast average is {abs(exec_change):.1f}% {'above' if exec_change >= 0 else 'below'} recent average demand.",
                    f"Inventory planning indicator: {exec_action}.",
                    "Check model error and investigate unusual historical periods before large purchasing decisions."
                ]:
                    st.markdown(f"- {item}")

                exec_summary = pd.DataFrame([{
                    "Target":str(selected_target),"Average Forecast":exec_avg,
                    "Total Forecast":exec_total,"Forecast Periods":len(exec_forecast),
                    "Recent Average Demand":exec_recent,"Forecast Change %":exec_change,
                    "Best Model":str(exec_model),"MAPE %":float(exec_mape) if exec_mape is not None else np.nan,
                    "Inventory Action":exec_action,"Recommended Horizon Stock":exec_stock
                }])
                st.download_button("⬇️ Download Executive Summary CSV", exec_summary.to_csv(index=False), "smartdemand_executive_summary.csv", "text/csv", use_container_width=True)

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")

st.markdown(
    '<div class="footer"><b>SmartDemand</b> • Demand Forecasting & Time Series Analysis<br>'
    'Built for data-driven business planning and decision support.</div>',
    unsafe_allow_html=True
)
