import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def create_bar_chart(csv_path, columns, output_dir):
    """
    Reads a CSV file, creates a bar chart for the counts of unique entries
    in each specified column, and saves them to output_dir/charts/.
    """
    charts_dir = os.path.join(output_dir, "charts")
    if not os.path.exists(charts_dir):
        os.makedirs(charts_dir)
        
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return

    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in {csv_path}. Skipping.")
            continue
            
        counts = df[col].value_counts()
        
        plt.figure(figsize=(10, 6))
        counts.plot(kind='bar')
        plt.title(f"Counts of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        
        output_file = os.path.join(charts_dir, f"{col}_bar_chart.png")
        plt.savefig(output_file)
        plt.close()
        print(f"Saved bar chart for '{col}' to {output_file}")

def create_model_year_chart(df_path, *, include_year=False):
    """
    Reads the cleaned complaints parquet file and creates a horizontal bar chart
    showing the top vehicle combinations by complaint count.

    Parameters
    ----------
    df_path : str
        Path to the cleaned complaints parquet file.
    include_year : bool, keyword-only, default False
        When False (default), groups by Make + Model only.
        When True, groups by Make + Model + Year, giving finer granularity
        at the cost of more bars and smaller counts per bar.

    Columns used (by position):
      - Index 3: Make  (e.g. "CHEVROLET")
      - Index 4: Model (e.g. "TRAILBLAZER")
      - Index 5: Year  (e.g. 1999)  — only used when include_year=True

    Each row in the parquet is one complaint. Rows that share the same
    key are grouped and counted. The chart shows the top 30 combinations
    so the chart stays readable.

    Design follows Tufte / Few principles:
      - Horizontal bars so long labels are legible
      - Bars sorted descending (most complaints at top) — preattentive ranking
      - Direct count labels at bar ends (eliminates axis interpolation)
      - Minimal non-data ink: no top/right spines, no fill color behind axes
      - Single neutral color; no legend needed (one data series)
      - Subtle vertical reference lines only at round tick values

    Intended use: call this function from Charts.ipynb. The chart is rendered
    inline by Jupyter via plt.show() — no file is saved.
    """
    # --- load data --------------------------------------------------------
    try:
        df = pd.read_parquet(df_path)
    except Exception as e:
        print(f"Error reading {df_path}: {e}")
        return

    # Grab make, model, and year by column position (0-indexed 3, 4, 5).
    # Using positional access makes this robust to column-name changes.
    make_col  = df.columns[3]
    model_col = df.columns[4]
    year_col  = df.columns[5]

    # Build a single "MAKE MODEL YEAR" label per row, then count occurrences.

    # Why the year column needs special handling:
    #   _normalize_list_columns in Build_DF.py converts YEARTXT to list<int64>
    #   so pyarrow gets a uniform column type. If even one collapsed group had
    #   conflicting year values, every row ends up as a list (e.g. [1999]).
    #   pd.to_numeric / astype can't handle list cells, so we must unwrap first.
    #
    # Also: NHTSA uses 9999 as an explicit "unknown model year" sentinel.
    def _extract_year(val) -> str:
        # pyarrow reads list<int64> columns as numpy arrays, not Python lists.
        # We unwrap both types the same way: grab the first element (or bail if
        # empty). val.flat[0] is used for numpy arrays because it works for both
        # 0-dimensional arrays (ndim==0, exactly one element) and 1-d arrays,
        # and it always returns a Python-compatible scalar.
        if isinstance(val, np.ndarray):
            if val.size == 0:
                return "UNKNOWN"
            val = val.flat[0]
        elif isinstance(val, list):
            if not val:
                return "UNKNOWN"
            val = val[0]
        # pd.isna handles None, np.nan, pd.NA, and pd.NaT uniformly.
        if pd.isna(val):
            return "UNKNOWN"
        year = int(val)
        # 9999 is the NHTSA sentinel for "model year unknown or N/A".
        return "UNKNOWN" if year == 9999 else str(year)

    _make_model = (
        df[make_col].astype(str).str.strip().str.upper()
        + " "
        + df[model_col].astype(str).str.strip().str.upper()
    )

    if include_year:
        _year_str = df[year_col].apply(_extract_year)
        df["_vehicle_key"] = _make_model + " " + _year_str
    else:
        df["_vehicle_key"] = _make_model

    # Count how many complaints each Make/Model/Year combination received,
    # then keep only the top 30 to preserve chart legibility.
    TOP_N = 30
    counts = df["_vehicle_key"].value_counts().head(TOP_N)

    # Reverse so the bar with the highest count appears at the top of the chart.
    counts = counts.iloc[::-1]

    # --- build chart ------------------------------------------------------
    # Height scales with number of bars so labels never overlap.
    fig_height = max(8, TOP_N * 0.35)
    _, ax = plt.subplots(figsize=(13, fig_height))

    # Horizontal bars — the correct choice for long categorical labels (Few, 2004).
    bars = ax.barh(
        counts.index,
        counts.values,
        color="#4C72B0",   # muted blue; professional and accessible
        edgecolor="none",  # remove bar outlines to reduce ink
        height=0.65,       # slight gap between bars aids separation
    )

    # Direct count labels at the right end of each bar.
    # This lets the reader read exact values without interpolating the axis
    # — a core Tufte principle (maximise data-ink, minimise redundant decoding).
    x_max = counts.values.max()
    for bar, val in zip(bars, counts.values):
        ax.text(
            val + x_max * 0.005,          # small horizontal offset from bar tip
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            va="center",
            ha="left",
            fontsize=8,
            color="#333333",
        )

    # --- axis & labels ----------------------------------------------------
    ax.set_xlabel("Number of Complaints", labelpad=8, fontsize=11)
    subtitle = "Make · Model · Year" if include_year else "Make · Model"
    ax.set_title(
        f"Top {TOP_N} Vehicles by Complaint Count\n({subtitle})",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    # Extend x-axis slightly beyond the max bar so labels aren't clipped.
    ax.set_xlim(0, x_max * 1.12)

    # Subtle vertical gridlines at round tick positions only (Tufte: let data stand out).
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#cccccc", alpha=0.7)
    ax.set_axisbelow(True)   # keep gridlines behind bars

    # Remove the top and right spines (chartjunk — Tufte).
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)   # y-labels already identify each bar

    # Tick formatting
    ax.tick_params(axis="y", labelsize=9, left=False)   # no tick marks on y-axis
    ax.tick_params(axis="x", labelsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    plt.show()