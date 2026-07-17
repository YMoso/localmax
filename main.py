import heapq

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


MAX_PLOT_POINTS = 1500
TOP_PEAKS_TO_SHOW = 20
MAX_EDITABLE_ROWS = 500


def fmt_num(value):
    if float(value).is_integer():
        return str(int(value))
    return str(round(float(value), 4))


def parse_numbers_from_text(text):
    parts = text.replace(",", " ").replace(";", " ").split()
    numbers = []

    for part in parts:
        try:
            numbers.append(float(part))
        except ValueError:
            pass

    return numbers


def create_200_point_example():
    """Create a repeatable 200-point sequence with three clear peaks."""
    x = np.linspace(0.0, 1.0, 200)
    values = (
        8.0
        + 0.8 * np.sin(8.0 * np.pi * x)
        + 10.0 * np.exp(-((x - 0.20) ** 2) / (2.0 * 0.035**2))
        + 16.0 * np.exp(-((x - 0.52) ** 2) / (2.0 * 0.055**2))
        + 12.0 * np.exp(-((x - 0.82) ** 2) / (2.0 * 0.040**2))
    )
    return np.round(values, 3).tolist()


def edit_numbers_table(numbers, key):
    if len(numbers) >= MAX_EDITABLE_ROWS:
        st.info(
            f"{len(numbers)} values loaded. Table editing is disabled for large datasets."
        )
        return numbers

    edited_df = st.data_editor(
        pd.DataFrame({"number": numbers}),
        num_rows="dynamic",
        hide_index=False,
        column_config={
            "number": st.column_config.NumberColumn(
                "Number",
                help="Edit, add, or delete values",
                required=False,
            )
        },
        key=key,
    )

    return edited_df["number"].dropna().astype(float).tolist()


def peak_key(peak):
    return (
        peak["strength"],
        peak["value"],
        peak["group_size"],
        -peak["matrix_index"],
    )


@st.cache_data(show_spinner=False, max_entries=3)
def find_all_peaks(numbers_tuple):
    numbers = np.asarray(numbers_tuple, dtype=float)
    number_count = len(numbers)
    cumulative_sum = np.concatenate(([0.0], np.cumsum(numbers)))
    top_heap = []
    sequence_number = 0

    # Larger groups have fewer than three rolling windows and cannot form a peak.
    for group_size in range(1, number_count - 1):
        window_sums = cumulative_sum[group_size:] - cumulative_sum[:-group_size]
        squared_sums = window_sums * window_sums

        previous_values = squared_sums[:-2]
        current_values = squared_sums[1:-1]
        next_values = squared_sums[2:]

        local_peak_mask = (
            (current_values > previous_values)
            & (current_values > next_values)
        )
        relative_indices = np.flatnonzero(local_peak_mask)

        if relative_indices.size == 0:
            continue

        matrix_indices = relative_indices + 1
        peak_values = current_values[relative_indices]
        strengths = (
            peak_values - previous_values[relative_indices]
            + peak_values - next_values[relative_indices]
        )

        candidate_count = min(TOP_PEAKS_TO_SHOW, relative_indices.size)
        candidate_order = np.lexsort(
            (
                matrix_indices,
                -peak_values,
                -strengths,
            )
        )[:candidate_count]

        for candidate_position in candidate_order:
            matrix_index = int(matrix_indices[candidate_position])
            peak = {
                "group": f"group_{group_size}",
                "group_size": group_size,
                "matrix_index": matrix_index,
                "value": float(peak_values[candidate_position]),
                "previous": float(squared_sums[matrix_index - 1]),
                "next": float(squared_sums[matrix_index + 1]),
                "strength": float(strengths[candidate_position]),
                "input_start_index": matrix_index,
                "input_end_index": matrix_index + group_size - 1,
            }
            heap_item = (peak_key(peak), sequence_number, peak)
            sequence_number += 1

            if len(top_heap) < TOP_PEAKS_TO_SHOW:
                heapq.heappush(top_heap, heap_item)
            elif heap_item[0] > top_heap[0][0]:
                heapq.heapreplace(top_heap, heap_item)

    if not top_heap:
        return None, []

    top_peaks = [
        item[2]
        for item in sorted(top_heap, key=lambda item: item[0], reverse=True)
    ]
    return top_peaks[0], top_peaks


def create_top_peaks_table(peaks):
    rows = []

    for rank, peak in enumerate(peaks, start=1):
        rows.append(
            {
                "Rank": rank,
                "Group": peak["group"],
                "Group size": peak["group_size"],
                "Peak index": peak["matrix_index"],
                "Input start": peak["input_start_index"],
                "Input end": peak["input_end_index"] + 1,
                "Peak value": round(peak["value"], 3),
                "Strength": round(peak["strength"], 3),
            }
        )

    return pd.DataFrame(rows)


def downsample_points(x_values, y_values, max_points=MAX_PLOT_POINTS):
    if len(x_values) <= max_points:
        return x_values, y_values

    step = int(np.ceil(len(x_values) / max_points))
    return x_values[::step], y_values[::step]


def plot_input_with_peak_range(numbers, peak):
    start = peak["input_start_index"]
    end = peak["input_end_index"]
    display_end = end + 1

    x_values = np.arange(len(numbers))
    raw_values = np.asarray(numbers, dtype=float)

    plot_x, plot_y = downsample_points(x_values, raw_values)
    selected_x = np.arange(start, display_end + 1)
    selected_y = raw_values[start:display_end + 1]

    y_min = float(raw_values.min())
    y_max = float(raw_values.max())
    y_range = y_max - y_min or 1.0
    baseline = y_min - y_range * 0.04

    fig, ax = plt.subplots(figsize=(14, 6), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(
        plot_x,
        plot_y,
        color="#4472C4",
        linewidth=1.2 if len(numbers) > MAX_PLOT_POINTS else 2.2,
        alpha=0.72,
        antialiased=True,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=2,
    )
    ax.fill_between(
        selected_x,
        selected_y,
        baseline,
        color="#ED7D31",
        alpha=0.18,
        interpolate=True,
        zorder=1,
    )
    ax.plot(
        selected_x,
        selected_y,
        color="#ED7D31",
        linewidth=1.8,
        antialiased=True,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=3,
    )

    for boundary in (start, display_end):
        ax.axvline(
            boundary,
            color="#A6A6A6",
            linewidth=1.0,
            linestyle=(0, (4, 4)),
            alpha=0.65,
            zorder=1,
        )

    ax.set_title(
        "Detected Peak in the Input Sequence",
        fontsize=18,
        fontweight="semibold",
        color="#262626",
        pad=18,
    )
    ax.set_xlabel("Input index", fontsize=14, color="#404040", labelpad=10)
    ax.set_ylabel("Input value", fontsize=14, color="#404040", labelpad=10)
    ax.tick_params(axis="both", labelsize=11, colors="#595959", length=0)
    ax.set_xlim(-0.5, len(numbers) - 0.5)
    ax.set_ylim(y_min - y_range * 0.08, y_max + y_range * 0.10)
    ax.grid(True, axis="both", color="#D9D9D9", linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color("#BFBFBF")
        spine.set_linewidth(0.8)

    fig.tight_layout()
    return fig


st.set_page_config(page_title="LocalMax", layout="wide")

tab_10, tab_200, tab_paste, tab_csv = st.tabs(
    ["10-value example", "200-point example", "Paste values", "Upload CSV"],
    default="10-value example",
    key="input_tab",
    on_change="rerun",
)

numbers = []

if tab_10.open:
    with tab_10:
        example_10 = [
            1.0,
            2.0,
            1.25,
            2.5,
            2.75,
            3.0,
            4.5,
            2.2,
            1.1,
            2.4,
        ]
        numbers = edit_numbers_table(example_10, key="example_10_editor")

elif tab_200.open:
    with tab_200:
        numbers = edit_numbers_table(
            create_200_point_example(),
            key="example_200_editor",
        )

elif tab_paste.open:
    with tab_paste:
        pasted_text = st.text_area(
            "Paste numbers",
            height=220,
            placeholder="1.5\n2.0\n1.25\n2.5\n2.75",
        )
        parsed_numbers = parse_numbers_from_text(pasted_text)

        if not parsed_numbers:
            st.info("Paste numbers to start.")
            st.stop()

        numbers = edit_numbers_table(parsed_numbers, key="paste_editor")

elif tab_csv.open:
    with tab_csv:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file is None:
            st.info("Upload a CSV file with a numeric column.")
            st.stop()

        uploaded_df = pd.read_csv(uploaded_file)
        numeric_columns = uploaded_df.select_dtypes(include=["number"]).columns.tolist()

        if not numeric_columns:
            st.error("No numeric columns found in the CSV file.")
            st.stop()

        selected_column = st.selectbox("Choose numeric column", numeric_columns)
        loaded_numbers = uploaded_df[selected_column].dropna().astype(float).tolist()
        numbers = edit_numbers_table(loaded_numbers, key="csv_editor")

if len(numbers) < 3:
    st.warning("Please enter at least 3 numbers.")
    st.stop()

candidate_count = (len(numbers) - 2) * (len(numbers) - 1) // 2
st.caption(
    f"Checking all {len(numbers) - 2:,} peak-producing group sizes "
    f"and {candidate_count:,} candidate positions."
)

with st.spinner("Checking all group sizes..."):
    best_peak, all_peaks = find_all_peaks(tuple(numbers))

if best_peak is None:
    st.warning("No local peak was found.")
else:
    fig_range = plot_input_with_peak_range(numbers, best_peak)
    st.pyplot(fig_range, clear_figure=True)
    plt.close(fig_range)

    with st.expander("Show top detected peaks"):
        st.dataframe(create_top_peaks_table(all_peaks), hide_index=True)
