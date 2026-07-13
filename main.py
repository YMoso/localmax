import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


MAX_AUTO_GROUP_SIZE = 500
MAX_PLOT_POINTS = 1500
TOP_PEAKS_TO_SHOW = 20
MAX_EDITABLE_ROWS = 200


def fmt_num(x):
    if float(x).is_integer():
        return str(int(x))
    return str(round(float(x), 4))


def parse_numbers_from_text(text):
    text = text.replace(",", " ")
    text = text.replace(";", " ")
    parts = text.split()

    numbers = []

    for part in parts:
        try:
            numbers.append(float(part))
        except ValueError:
            pass

    return numbers


def edit_numbers_table(numbers, key):
    if len(numbers) >= MAX_EDITABLE_ROWS:
        st.info(
            f"{len(numbers)} values loaded. Table editing is disabled for large datasets."
        )
        return numbers

    df = pd.DataFrame({"number": numbers})

    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        hide_index=False,
        column_config={
            "number": st.column_config.NumberColumn(
                "Number",
                help="Edit, add, or delete values",
                required=False
            )
        },
        key=key
    )

    return edited_df["number"].dropna().astype(float).tolist()


def get_auto_group_size(numbers):
    return min(len(numbers), MAX_AUTO_GROUP_SIZE)


def peak_key(peak):
    return (
        peak["strength"],
        peak["value"],
        peak["group_size"],
        -peak["matrix_index"]
    )


def find_all_peaks(numbers):
    top_peaks = []
    best_in_groups = {}

    arr = np.array(numbers, dtype=float)
    max_group_size = get_auto_group_size(numbers)

    cumsum = np.concatenate(([0], np.cumsum(arr)))

    for group_size in range(1, max_group_size + 1):
        window_sums = cumsum[group_size:] - cumsum[:-group_size]
        squared_sums = np.round(window_sums ** 2, 3)

        if len(squared_sums) < 3:
            continue

        previous_values = squared_sums[:-2]
        current_values = squared_sums[1:-1]
        next_values = squared_sums[2:]

        peak_mask = (current_values > previous_values) & (current_values > next_values)
        peak_indices = np.where(peak_mask)[0] + 1

        group_best_peak = None

        for i in peak_indices:
            previous_value = squared_sums[i - 1]
            current_value = squared_sums[i]
            next_value = squared_sums[i + 1]

            strength = (current_value - previous_value) + (current_value - next_value)

            peak = {
                "group": f"group_{group_size}",
                "group_size": group_size,
                "matrix_index": int(i),
                "value": float(current_value),
                "previous": float(previous_value),
                "next": float(next_value),
                "strength": float(strength),
                "input_start_index": int(i),
                "input_end_index": int(i + group_size - 1)
            }

            if group_best_peak is None or peak_key(peak) > peak_key(group_best_peak):
                group_best_peak = peak

            top_peaks.append(peak)

        if group_best_peak is not None:
            best_in_groups[f"group_{group_size}"] = group_best_peak

        if len(top_peaks) > TOP_PEAKS_TO_SHOW * 5:
            top_peaks = sorted(
                top_peaks,
                key=peak_key,
                reverse=True
            )[:TOP_PEAKS_TO_SHOW]

    if not top_peaks:
        return None, [], best_in_groups

    top_peaks = sorted(
        top_peaks,
        key=peak_key,
        reverse=True
    )[:TOP_PEAKS_TO_SHOW]

    best_peak = top_peaks[0]

    return best_peak, top_peaks, best_in_groups


def create_top_peaks_table(peaks):
    rows = []

    for peak in peaks[:TOP_PEAKS_TO_SHOW]:
        start = peak["input_start_index"]
        end = peak["input_end_index"]
        display_end = end + 1

        rows.append({
            "Rank": len(rows) + 1,
            "Group": peak["group"],
            "Group size": peak["group_size"],
            "Peak index": peak["matrix_index"],
            "Input start": start,
            "Input end": display_end,
            "Peak value": round(float(peak["value"]), 3),
            "Strength": round(float(peak["strength"]), 3)
        })

    return pd.DataFrame(rows)


def downsample_points(x, y, max_points=1500):
    if len(x) <= max_points:
        return x, y

    step = int(np.ceil(len(x) / max_points))

    sampled_x = x[::step]
    sampled_y = y[::step]

    return sampled_x, sampled_y


def plot_input_with_peak_range(numbers, peak):
    start = peak["input_start_index"]
    end = peak["input_end_index"]
    display_end = end + 1

    full_x = list(range(len(numbers)))
    full_y = numbers

    plot_x, plot_y = downsample_points(
        full_x,
        full_y,
        max_points=MAX_PLOT_POINTS
    )

    selected_x = list(range(start, display_end + 1))
    selected_y = numbers[start:end + 1] + [numbers[end]]

    y_min = min(numbers)
    y_max = max(numbers)
    y_range = y_max - y_min

    if y_range == 0:
        y_range = 1

    baseline = y_min - y_range * 0.08

    plt.rcParams["font.family"] = "DejaVu Sans"

    fig, ax = plt.subplots(figsize=(15, 6), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if len(numbers) <= MAX_PLOT_POINTS:
        ax.step(
            full_x,
            full_y,
            where="post",
            linewidth=1.0,
            color="#64748b",
            alpha=0.38
        )
    else:
        ax.step(
            plot_x,
            plot_y,
            where="post",
            linewidth=1.0,
            color="#64748b",
            alpha=0.22
        )

    ax.fill_between(
        selected_x,
        selected_y,
        baseline,
        step="post",
        color="#f59e0b",
        alpha=0.26
    )

    ax.step(
        selected_x,
        selected_y,
        where="post",
        linewidth=0.3,
        color="#b45309"
    )

    ax.axvline(
        start,
        linewidth=1.2,
        linestyle="--",
        color="#92400e",
        alpha=0.25
    )

    ax.axvline(
        display_end,
        linewidth=1.2,
        linestyle="--",
        color="#92400e",
        alpha=0.25
    )

    ax.set_title(
        "Detected Peak in the Input Sequence",
        fontsize=21,
        fontweight="bold",
        pad=22
    )

    ax.set_xlabel(
        "Input index",
        fontsize=17,
        fontweight="bold",
        labelpad=14
    )

    ax.set_ylabel(
        "Input value",
        fontsize=17,
        fontweight="bold",
        labelpad=14
    )

    ax.tick_params(
        axis="both",
        labelsize=13,
        width=1.3,
        length=6,
        colors="#1f2937"
    )

    ax.set_xlim(-0.5, len(numbers) - 0.5)
    ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.15)

    ax.grid(
        True,
        axis="y",
        color="#e5e7eb",
        linewidth=1.1
    )

    ax.grid(False, axis="x")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.spines["left"].set_color("#1f2937")
    ax.spines["bottom"].set_color("#1f2937")

    fig.tight_layout()

    return fig


st.set_page_config(page_title="LocalMax", layout="wide")

input_method = st.radio(
    "Input method",
    ["Manual table", "Paste values", "Upload CSV"],
    horizontal=True,
    label_visibility="collapsed"
)

numbers = []

if input_method == "Paste values":
    pasted_text = st.text_area(
        "Paste numbers",
        height=220,
        placeholder="1.5\n2.0\n1.25\n2.5\n2.75"
    )

    parsed_numbers = parse_numbers_from_text(pasted_text)

    if not parsed_numbers:
        st.info("Paste numbers to start.")
        st.stop()

    numbers = edit_numbers_table(parsed_numbers, key="paste_editor")

elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is None:
        st.info("Upload a CSV file with a numeric column.")
        st.stop()

    uploaded_df = pd.read_csv(uploaded_file)

    numeric_columns = uploaded_df.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_columns:
        st.error("No numeric columns found in the CSV file.")
        st.stop()

    selected_column = st.selectbox(
        "Choose numeric column",
        numeric_columns
    )

    loaded_numbers = uploaded_df[selected_column].dropna().astype(float).tolist()

    if len(loaded_numbers) >= MAX_EDITABLE_ROWS:
        st.info(f"{len(loaded_numbers)} values loaded.")
        numbers = loaded_numbers
    else:
        numbers = edit_numbers_table(loaded_numbers, key="csv_editor")

else:
    default_numbers = [
        1.0, 2.0, 1.25, 2.5, 2.75,
        3.0, 4.5, 2.2, 1.1, 2.4
    ]

    numbers = edit_numbers_table(default_numbers, key="manual_editor")


if len(numbers) < 3:
    st.warning("Please enter at least 3 numbers.")
    st.stop()


with st.spinner("Finding peaks..."):
    best_peak, all_peaks, best_by_group = find_all_peaks(numbers)


if best_peak is None:
    st.warning("No local peak was found.")
else:
    fig_range = plot_input_with_peak_range(numbers, best_peak)
    st.pyplot(fig_range, clear_figure=True)
    plt.close(fig_range)

    peaks_table = create_top_peaks_table(all_peaks)

    with st.expander("Show top detected peaks"):
        st.dataframe(peaks_table, hide_index=True)