import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def _is_missing_cell(value):
    """Safely detect missing scalar/list-like parquet cells."""
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, list):
        return len(value) == 0
    try:
        return pd.isna(value)
    except (TypeError, ValueError):
        return False


def _cell_matches_filter(value, expected):
    """Return True when a scalar or list-like cell matches a filter value."""
    expected_values = expected if isinstance(expected, list) else [expected]
    expected_strings = {str(v).strip().upper() for v in expected_values}

    if isinstance(value, np.ndarray):
        actual_values = value.tolist()
    elif isinstance(value, list):
        actual_values = value
    else:
        actual_values = [value]

    actual_strings = {str(v).strip().upper() for v in actual_values}
    return bool(actual_strings & expected_strings)


def _apply_chart_filters(df, filters):
    """Apply simple equality/member filters used by notebook and LangGraph charts."""
    if not filters:
        return df

    filtered = df
    for col, expected in filters.items():
        if col not in filtered.columns:
            print(f"Warning: Column '{col}' not found in dataframe. Skipping filter.")
            continue
        filtered = filtered[filtered[col].apply(lambda value: _cell_matches_filter(value, expected))]
        if filtered.empty:
            break
    return filtered


def _flatten_component_labels(series):
    """Flatten COMPDESC cells into one clean label per human component assignment."""
    labels = []
    for value in series:
        if _is_missing_cell(value):
            continue
        if isinstance(value, np.ndarray):
            values = value.tolist()
        elif isinstance(value, list):
            values = value
        else:
            values = [value]

        for label in values:
            if _is_missing_cell(label):
                continue
            cleaned = str(label).strip().upper()
            if cleaned:
                labels.append(cleaned)
    return labels


def create_human_subsystem_frequency_chart(df_path, filters=None, top_n=15):
    """
    Plot the most frequent human-labelled subsystem components from COMPDESC.

    Parameters
    ----------
    df_path : str
        Path to the cleaned complaints parquet file.
    filters : dict, optional
        Column-to-value filters derived from the user's query. List-like columns
        such as COMPDESC match when the requested value appears in the cell.
        Example: {"MAKETXT": "TOYOTA", "CRASH": "Y"}.
    top_n : int, default 15
        Number of most frequent human component labels to display.

    Returns
    -------
    matplotlib.figure.Figure | None
        The rendered figure, or None if the data cannot be loaded or no labels
        remain after filtering.
    """
    try:
        df = pd.read_parquet(df_path)
    except Exception as e:
        print(f"Error reading {df_path}: {e}")
        return

    filters = filters or {}
    top_n = max(1, min(int(top_n), 30))

    filtered = _apply_chart_filters(df, filters)
    if filtered.empty:
        print("No matching complaints found for the given filters.")
        return

    if "COMPDESC" not in filtered.columns:
        print("COMPDESC column not found; cannot chart human subsystem labels.")
        return

    labels = _flatten_component_labels(filtered["COMPDESC"])
    if not labels:
        print("No human subsystem labels found after filtering.")
        return

    counts = pd.Series(labels).value_counts().head(top_n)

    fig_width = max(12, min(20, top_n * 0.9))
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    bars = ax.bar(
        counts.index,
        counts.values,
        color="#4C72B0",
        edgecolor="none",
        width=0.72,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.6, color="#EAEAEA", alpha=0.9)
    ax.set_axisbelow(True)

    ax.set_xlabel("Human Labelled Subsystem Component", fontsize=11, labelpad=10)
    ax.set_ylabel("Number of Complaints", fontsize=11, labelpad=10)

    filter_text = "All complaints" if not filters else "Filtered complaints"
    ax.set_title(
        f"Top {len(counts)} Human Labelled Subsystem Components\n"
        f"{filter_text} (n={len(filtered):,} complaints)",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )

    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    for label in ax.get_xticklabels():
        label.set_ha("right")

    y_max = counts.values.max()
    ax.set_ylim(0, y_max * 1.12)
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + y_max * 0.015,
            f"{int(value):,}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )

    plt.tight_layout()
    plt.show()
    return fig

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

    # Track the last figure created so callers (e.g. Chart_Tools.py) can
    # capture it via the return value. Each loop iteration creates a new figure,
    # so we grab the current figure with plt.gcf() right before plt.close().
    last_fig = None

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
        # Capture the figure object BEFORE closing — Chart_Tools.py uses the
        # returned fig to render to an in-memory PNG (Agg backend) rather than
        # saving to disk. We store only the last figure from the loop; callers
        # that need a specific column should pass a single-element columns list.
        last_fig = plt.gcf()
        plt.close()
        print(f"Saved bar chart for '{col}' to {output_file}")

    return last_fig

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
    # Capture the figure object before plt.show() so Chart_Tools.py can grab it.
    # plt.subplots() above returned (fig, ax) but discarded fig via the _ pattern;
    # plt.gcf() retrieves the same object from matplotlib's internal state tracker.
    fig = plt.gcf()
    plt.show()
    return fig


def compare_LLM_to_NHTSA(csv_path, parquet_path):
    """
    Compares LLM-predicted subsystem labels against NHTSA expert (COMPDESC) labels
    for the sampled complaints stored in csv_path.

    For each complaint in the CSV, this function:
      1. Parses the LLM's subsystem predictions (stored as a Python list string).
      2. Looks up the corresponding NHTSA COMPDESC labels from the parquet file
         using the stored DataFrame index (the 'df_index' column).
      3. Classifies each complaint into one of three agreement categories.

    Agreement definitions (all comparisons are uppercase + stripped):
      - A "match" between an LLM label and an NHTSA label occurs when:
          * The strings are exactly equal, OR
          * The NHTSA label begins with the LLM label followed by ':' or '/'
            (accounts for NHTSA's hierarchical sub-category naming, e.g. the LLM
            predicts 'FORWARD COLLISION AVOIDANCE' and NHTSA has the more specific
            'FORWARD COLLISION AVOIDANCE: WARNINGS'. The ':' / '/' guard prevents
            false prefix matches like 'ENGINE' incorrectly matching
            'ENGINE AND ENGINE COOLING').
      - Full Agreement:    The LLM's label set equals the NHTSA set exactly
                          (every label has an exact match, same cardinality).
      - Partial Agreement: At least one LLM label has a match in the NHTSA set
                          (exact or prefix), but the sets are not identical.
      - No Agreement:     No match of any kind between the two sets.

    Renders three subplots inline via plt.show() — no file is saved:
      1. Overall agreement counts (Full / Partial / No Agreement) with percentages.
      2. Per-NHTSA-category accuracy: for the top 20 most-frequent NHTSA categories,
         stacked bars show how many complaint-label pairs the LLM correctly identified
         vs missed. Each NHTSA label is evaluated independently, so a complaint with
         two NHTSA labels contributes two data points to this subplot.
      3. Miss confusion heatmap: for every "No Agreement" complaint, every possible
         (NHTSA label, LLM label) cross-product pair is recorded. The heatmap shows
         how frequently each (NHTSA, LLM) combination appeared, making it easy to
         see which LLM labels get substituted for which NHTSA labels when the LLM
         completely misses. Darker cells = more frequent confusion.

    Parameters
    ----------
    csv_path : str
        Path to the LLM output CSV. Required columns: 'df_index', 'subsystems'.
        'subsystems' must be a string representation of a Python list
        (e.g. "['ENGINE', 'POWER TRAIN']").
    parquet_path : str
        Path to the cleaned NHTSA complaints parquet file. Column 11 (COMPDESC)
        must contain the expert-labelled subsystem numpy arrays.
    """
    import ast
    from collections import Counter
    from matplotlib.gridspec import GridSpec

    # --- load data --------------------------------------------------------
    try:
        csv_df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return

    try:
        pq_df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Error reading {parquet_path}: {e}")
        return

    # --- helpers ----------------------------------------------------------

    def _normalize(labels):
        """Convert an iterable of label strings to a frozenset of uppercased,
        stripped strings for comparison."""
        return frozenset(str(l).upper().strip() for l in labels)

    def _llm_label_matches_nhtsa_set(llm_label, nhtsa_set):
        """
        Returns True if llm_label has any match in nhtsa_set.

        Match criteria:
          - Exact equality.
          - NHTSA label starts with llm_label + ':' or llm_label + '/'
            (handles hierarchical NHTSA sub-categories like
            'FORWARD COLLISION AVOIDANCE: WARNINGS' being correctly covered
            by the LLM's coarser 'FORWARD COLLISION AVOIDANCE').
        """
        for n in nhtsa_set:
            if llm_label == n:
                return True
            if n.startswith(llm_label + ":") or n.startswith(llm_label + "/"):
                return True
        return False

    def _nhtsa_label_matched_by_llm(nhtsa_label, llm_set):
        """
        Returns True if the LLM produced a label that covers nhtsa_label.
        Mirrors the same prefix logic as _llm_label_matches_nhtsa_set but
        from the NHTSA label's perspective (used for per-category subplot).
        """
        for l in llm_set:
            if nhtsa_label == l:
                return True
            if nhtsa_label.startswith(l + ":") or nhtsa_label.startswith(l + "/"):
                return True
        return False

    # --- classify each complaint & build data for all three subplots ------
    full_agree    = 0
    partial_agree = 0
    no_agree      = 0

    # Subplot 2: one entry per (complaint, NHTSA label) pair.
    per_label_records = []   # list of (nhtsa_label: str, matched: bool)

    # Subplot 3: for every "No Agreement" complaint, record every possible
    # (nhtsa_label, llm_label) cross-product pair. Counting these pairs
    # reveals which LLM labels get substituted for which NHTSA labels when
    # the LLM completely misses.
    miss_pairs = []          # list of (nhtsa_label: str, llm_label: str)

    # Full-disagreement cases: stored as (df_index, complaint_text) so they can
    # be printed after the charts for manual inspection.
    no_agree_cases = []      # list of (df_index: int, cdescr: str)

    for _, row in csv_df.iterrows():
        # LLM subsystems are stored as a Python list literal string in the CSV.
        try:
            llm_labels = ast.literal_eval(row["subsystems"])
        except (ValueError, SyntaxError):
            llm_labels = []

        llm_set   = _normalize(llm_labels)
        nhtsa_raw = pq_df.loc[row["df_index"], "COMPDESC"]
        nhtsa_set = _normalize(list(nhtsa_raw))

        # --- overall complaint classification ---
        if llm_set == nhtsa_set:
            full_agree += 1
        else:
            any_match = any(_llm_label_matches_nhtsa_set(l, nhtsa_set) for l in llm_set)
            if any_match:
                partial_agree += 1
            else:
                no_agree += 1
                # Record every (nhtsa, llm) cross-product for the heatmap.
                # We use all combinations because a single complaint can have
                # multiple NHTSA labels and multiple LLM labels simultaneously.
                for n in nhtsa_set:
                    for l in llm_set:
                        miss_pairs.append((n, l))
                # Store the complaint index, labels, and text for post-chart printing.
                complaint_text = str(pq_df.loc[row["df_index"], "CDESCR"])
                no_agree_cases.append((row["df_index"], nhtsa_set, llm_set, complaint_text))

        # --- per-label records (subplot 2, from the NHTSA label's perspective) ---
        for nhtsa_label in nhtsa_set:
            matched = _nhtsa_label_matched_by_llm(nhtsa_label, llm_set)
            per_label_records.append((nhtsa_label, matched))

    total = full_agree + partial_agree + no_agree

    # --- aggregate per-category stats for subplot 2 ----------------------
    per_label_df = pd.DataFrame(per_label_records, columns=["nhtsa_label", "matched"])
    cat_stats = (
        per_label_df
        .groupby("nhtsa_label")["matched"]
        .agg(total="count", agreed="sum")
        .reset_index()
    )
    cat_stats["missed"] = cat_stats["total"] - cat_stats["agreed"]

    TOP_N_CATS = 20
    cat_stats = cat_stats.sort_values("total", ascending=False).head(TOP_N_CATS)
    # Reverse so the highest-frequency category sits at the top of the chart.
    cat_stats = cat_stats.iloc[::-1].reset_index(drop=True)

    agreed_vals = cat_stats["agreed"].values
    missed_vals = cat_stats["missed"].values
    cat_labels  = cat_stats["nhtsa_label"].values

    # --- build pivot table for subplot 3 (miss confusion heatmap) --------
    # Count occurrences of each (nhtsa_label, llm_label) miss pair.
    pair_counts = Counter(miss_pairs)
    pair_rows   = [(n, l, c) for (n, l), c in pair_counts.items()]
    pair_df     = pd.DataFrame(pair_rows, columns=["nhtsa", "llm", "count"])

    # Pivot to a 2-D matrix: rows = NHTSA labels, columns = LLM labels.
    # Sort both axes by their total occurrence count (most-confused first).
    nhtsa_order = pair_df.groupby("nhtsa")["count"].sum().sort_values(ascending=False).index
    llm_order   = pair_df.groupby("llm")["count"].sum().sort_values(ascending=False).index
    pivot = (
        pair_df
        .pivot(index="nhtsa", columns="llm", values="count")
        .reindex(index=nhtsa_order, columns=llm_order)
        .fillna(0)
        .astype(int)
    )

    # --- build figure with GridSpec layout --------------------------------
    # Row 0: two side-by-side subplots (overall agreement + per-category).
    # Row 1: full-width heatmap spanning both columns.
    # height_ratios gives the heatmap room proportional to the number of
    # NHTSA rows it needs to display.
    top_height  = max(8, TOP_N_CATS * 0.5)
    heat_height = max(4, len(pivot.index) * 0.55)

    fig = plt.figure(figsize=(18, top_height + heat_height + 1))
    gs  = GridSpec(
        2, 2, figure=fig,
        height_ratios=[top_height, heat_height],
        hspace=0.55,   # vertical gap between rows
        wspace=0.35,   # horizontal gap between columns in row 0
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])   # spans both columns

    # ---- Subplot 1: Overall agreement breakdown -------------------------
    categories = ["Full Agreement", "Partial Agreement", "No Agreement"]
    counts     = [full_agree, partial_agree, no_agree]
    colors     = ["#2ca02c", "#ff7f0e", "#d62728"]

    bars1  = ax1.barh(categories, counts, color=colors, edgecolor="none", height=0.5)
    x1_max = max(counts) if max(counts) > 0 else 1
    for bar, val in zip(bars1, counts):
        pct = val / total * 100 if total > 0 else 0
        ax1.text(
            val + x1_max * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val}  ({pct:.1f}%)",
            va="center", ha="left", fontsize=10, color="#333333",
        )

    ax1.set_xlim(0, x1_max * 1.35)
    ax1.set_xlabel("Number of Complaints", labelpad=8, fontsize=11)
    ax1.set_title(
        f"LLM vs. NHTSA Expert Agreement\n(n={total} sampled complaints)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax1.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#cccccc", alpha=0.7)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.tick_params(axis="y", left=False, labelsize=10)
    ax1.tick_params(axis="x", labelsize=9)

    # ---- Subplot 2: Per-NHTSA-category accuracy -------------------------
    y_pos = range(len(cat_stats))
    ax2.barh(y_pos, agreed_vals, color="#2ca02c", edgecolor="none",
             height=0.6, label="LLM Agreed")
    ax2.barh(y_pos, missed_vals, left=agreed_vals, color="#d62728",
             edgecolor="none", height=0.6, label="LLM Missed")

    x2_max = (agreed_vals + missed_vals).max() if len(agreed_vals) > 0 else 1
    for i, (a, m) in enumerate(zip(agreed_vals, missed_vals)):
        total_bar = a + m
        pct = a / total_bar * 100 if total_bar > 0 else 0
        ax2.text(
            total_bar + x2_max * 0.01,
            i,
            f"{pct:.0f}%",
            va="center", ha="left", fontsize=8, color="#333333",
        )

    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels(cat_labels, fontsize=7)
    ax2.set_xlim(0, x2_max * 1.15)
    ax2.set_xlabel("Number of Complaint-Label Pairs", labelpad=8, fontsize=11)
    ax2.set_title(
        f"Per-Category Agreement\n(top {TOP_N_CATS} NHTSA categories by occurrence)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax2.legend(loc="lower right", fontsize=9, framealpha=0.8)
    ax2.xaxis.grid(True, linestyle="--", linewidth=0.5, color="#cccccc", alpha=0.7)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="y", left=False)
    ax2.tick_params(axis="x", labelsize=9)

    # ---- Subplot 3: Miss confusion heatmap ------------------------------
    # Each cell shows how many times the LLM predicted the column label
    # (x-axis) for a complaint whose actual NHTSA label was the row label
    # (y-axis), across all "No Agreement" complaints.
    # Darker blue = the LLM more frequently substituted that column label
    # for the true NHTSA row label.
    if pivot.empty:
        ax3.text(0.5, 0.5, "No 'No Agreement' cases to display.",
                 ha="center", va="center", fontsize=12, transform=ax3.transAxes)
        ax3.set_axis_off()
    else:
        heat_data = pivot.values.astype(float)
        # Replace 0 with NaN so empty cells render as white (no confusion there).
        heat_display = np.where(heat_data == 0, np.nan, heat_data)

        im = ax3.imshow(
            heat_display,
            cmap="Blues",
            aspect="auto",
            vmin=0,                        # anchor color scale at 0
            vmax=heat_data.max() or 1,     # top of scale = most-frequent confusion
        )

        # Annotate every non-zero cell with its count so exact values are readable
        # without needing a colorbar.
        for row_i in range(heat_data.shape[0]):
            for col_j in range(heat_data.shape[1]):
                val = int(heat_data[row_i, col_j])
                if val == 0:
                    continue
                # White text on darker cells, dark text on lighter cells.
                text_color = "white" if val >= heat_data.max() * 0.6 else "#333333"
                ax3.text(
                    col_j, row_i, str(val),
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_color,
                )

        ax3.set_xticks(range(len(pivot.columns)))
        ax3.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=8)
        ax3.set_yticks(range(len(pivot.index)))
        ax3.set_yticklabels(pivot.index, fontsize=8)

        # Label both axes so the reader knows which direction is "truth" vs "prediction".
        ax3.set_xlabel("LLM Predicted Label", labelpad=10, fontsize=11)
        ax3.set_ylabel("NHTSA Expert Label\n(ground truth)", labelpad=10, fontsize=11)
        ax3.set_title(
            f"Miss Confusion: What the LLM Said When It Completely Missed\n"
            f"({no_agree} 'No Agreement' complaints · cell = number of cross-product pairs)",
            fontsize=12, fontweight="bold", pad=12,
        )

        # Subtle colorbar on the right so the reader can gauge intensity.
        cbar = fig.colorbar(im, ax=ax3, shrink=0.6, pad=0.02)
        cbar.set_label("Pair count", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    plt.show()
    # compare_LLM_to_NHTSA explicitly builds `fig` via plt.figure() above,
    # so no plt.gcf() is needed here — just return it directly.
    # Chart_Tools.py uses this fig to render to an in-memory PNG.

    # --- print full-disagreement complaints --------------------------------
    # These are cases where the LLM's labels shared zero overlap with the
    # NHTSA expert labels — worth reviewing manually to understand failure modes.
    print(f"\n{'='*70}")
    print(f"FULL DISAGREEMENT CASES  ({len(no_agree_cases)} of {total} complaints)")
    print(f"{'='*70}\n")
    for idx, (df_idx, nhtsa_lbls, llm_lbls, text) in enumerate(no_agree_cases, start=1):
        print(f"[{idx}] Complaint index: {df_idx}")
        print(f"    Human labels: {', '.join(sorted(nhtsa_lbls))}")
        print(f"    LLM labels:   {', '.join(sorted(llm_lbls))}")
        print(f"    {text}")
        print()

    return fig
