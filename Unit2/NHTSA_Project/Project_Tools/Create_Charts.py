import pandas as pd
import matplotlib.pyplot as plt
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

def create_model_year_chart(df_path):
    """
    Reads the cleaned complaints parquet file and creates a horizontal bar chart
    showing the top vehicle Make+Model+Year combinations by complaint count.

    Columns used (by position):
      - Index 3: Make  (e.g. "CHEVROLET")
      - Index 4: Model (e.g. "TRAILBLAZER")
      - Index 5: Year  (e.g. 1999)

    Each row in the parquet is one complaint. Rows that share the same
    Make/Model/Year are grouped and counted. The chart shows the top 30
    combinations so the chart stays readable.

    Design follows Tufte / Few principles:
      - Horizontal bars so long Make-Model-Year labels are legible
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
    # Cast year to int to drop any decimal from float representation (e.g. 1999.0 → 1999).
    df["_vehicle_key"] = (
        df[make_col].astype(str).str.strip().str.upper()
        + " "
        + df[model_col].astype(str).str.strip().str.upper()
        + " "
        + df[year_col].apply(lambda y: str(int(float(y))) if pd.notna(y) else "UNKNOWN")
    )

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
    ax.set_title(
        f"Top {TOP_N} Vehicles by Complaint Count\n(Make · Model · Year)",
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