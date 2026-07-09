import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


def fmt_num(x):
    if float(x).is_integer():
        return str(int(x))
    return str(round(float(x), 4))


def fmt_table_value(x):
    if pd.isna(x):
        return ""

    if isinstance(x, (int, float, np.integer, np.floating)):
        return f"{float(x):.3f}".rstrip("0").rstrip(".")

    return str(x)


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


def make_matrix(numbers):
    n = len(numbers)
    arr = np.array(numbers, dtype=float)
    result = {}

    for size in range(1, n + 1):
        col_values = []

        for i in range(n - size + 1):
            total = arr[i:i + size].sum()
            squared_total = total ** 2
            col_values.append(round(squared_total, 3))

        result[f"group_{size}"] = pd.Series(col_values)

    return pd.DataFrame(result)


def find_peaks_for_column(values, group_size, column_name):
    peaks = []

    for i in range(1, len(values) - 1):
        prev_val = values[i - 1]
        curr_val = values[i]
        next_val = values[i + 1]

        if curr_val > prev_val and curr_val > next_val:
            strength = (curr_val - prev_val) + (curr_val - next_val)

            peaks.append({
                "column": column_name,
                "group_size": group_size,
                "matrix_index": i,
                "value": curr_val,
                "previous": prev_val,
                "next": next_val,
                "strength": strength,
                "input_start_index": i,
                "input_end_index": i + group_size - 1
            })

    return peaks


def find_all_peaks(matrix):
    all_peaks = []
    best_in_columns = {}

    for col in matrix.columns:
        vals = matrix[col].dropna().tolist()
        group_size = int(col.replace("group_", ""))

        peaks = find_peaks_for_column(vals, group_size, col)

        if len(peaks) > 0:
            best_peak = max(
                peaks,
                key=lambda p: (
                    p["strength"],
                    p["value"],
                    p["group_size"],
                    -p["matrix_index"]
                )
            )

            best_in_columns[col] = best_peak
            all_peaks.extend(peaks)

    if len(all_peaks) == 0:
        return None, best_in_columns

    best_overall = max(
        all_peaks,
        key=lambda p: (
            p["strength"],
            p["value"],
            p["group_size"],
            -p["matrix_index"]
        )
    )

    return best_overall, best_in_columns


def highlight_table(matrix, best_peak, best_by_col):
    style = pd.DataFrame("", index=matrix.index, columns=matrix.columns)

    for col, peak in best_by_col.items():
        row = peak["matrix_index"]

        style.loc[row, col] = (
            "background-color: #ffdd57; "
            "color: black; "
            "font-weight: bold;"
        )

    if best_peak is not None:
        row = best_peak["matrix_index"]
        col = best_peak["column"]

        style.loc[row, col] = (
            "background-color: #8fd19e; "
            "color: black; "
            "font-weight: bold;"
        )

    return style


def show_interval(numbers, peak):
    start = peak["input_start_index"]
    end = peak["input_end_index"]

    html = """
    <div style="
        display:flex;
        flex-wrap:wrap;
        gap:8px;
        margin-top:16px;
        margin-bottom:16px;
    ">
    """

    for i, num in enumerate(numbers):
        selected = start <= i <= end

        if selected:
            bg = "#d4edda"
            border = "#2f9e44"
            weight = "700"
        else:
            bg = "#f1f3f5"
            border = "#dee2e6"
            weight = "400"

        html += f"""
        <div style="
            padding:10px 14px;
            border-radius:10px;
            border:2px solid {border};
            background-color:{bg};
            color:black;
            font-weight:{weight};
            text-align:center;
            min-width:55px;
        ">
            <div style="font-size:12px; color:#666;">{i}</div>
            <div style="font-size:18px;">{fmt_num(num)}</div>
        </div>
        """

    html += "</div>"

    st.html(html)


def plot_input_with_peak_range(numbers, peak):
    start = peak["input_start_index"]
    end = peak["input_end_index"]

    x = list(range(len(numbers)))
    selected_x = x[start:end + 1]
    selected_y = numbers[start:end + 1]

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(
        x,
        numbers,
        marker="o",
        linewidth=1.4,
        markersize=4,
        alpha=0.55,
        label="All input values"
    )

    ax.plot(
        selected_x,
        selected_y,
        marker="o",
        linewidth=3,
        markersize=8,
        label="Selected peak range"
    )

    ax.axvspan(
        start,
        end,
        alpha=0.18
    )

    ax.axvline(
        start,
        linestyle="--",
        linewidth=1.5,
        alpha=0.8
    )

    ax.axvline(
        end,
        linestyle="--",
        linewidth=1.5,
        alpha=0.8
    )

    for i in range(start, end + 1):
        ax.annotate(
            fmt_num(numbers[i]),
            xy=(i, numbers[i]),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold"
        )

    ax.annotate(
        f"input[{start}]",
        xy=(start, numbers[start]),
        xytext=(0, -28),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        fontweight="bold"
    )

    ax.annotate(
        f"input[{end}]",
        xy=(end, numbers[end]),
        xytext=(0, -28),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        fontweight="bold"
    )

    ax.set_title(
        f"Selected peak range: {start} -- {end}",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Input index")
    ax.set_ylabel("Input value")

    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    return fig

def show_peak_details(numbers, peak):
    start = peak["input_start_index"]
    end = peak["input_end_index"]

    selected_numbers = numbers[start:end + 1]
    selected_sum = sum(selected_numbers)
    selected_squared_sum = selected_sum ** 2
    calculation = " + ".join(fmt_num(x) for x in selected_numbers)

    st.info(
        f"""
        Peak found in **{peak["column"]}**  
        Matrix index: **{peak["matrix_index"]}**  
        
        """
    )


st.set_page_config(page_title="LocalMax", layout="wide")

input_method = st.radio(
    "",
    ["Manual table", "Paste values", "Upload CSV"],
    horizontal=True
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


matrix = make_matrix(numbers)
best_peak, best_by_column = find_all_peaks(matrix)


if best_peak is None:
    st.info("No local peak was found, so there is no interval to show.")
else:
    start = best_peak["input_start_index"]
    end = best_peak["input_end_index"]
    chosen_numbers = numbers[start:end + 1]
    chosen_sum = sum(chosen_numbers)
    chosen_squared_sum = chosen_sum ** 2

    calculation = " + ".join(fmt_num(x) for x in chosen_numbers)



if best_peak is None:
    st.warning("No local peak was found.")
    st.dataframe(matrix.style.format(fmt_table_value))
else:
    styled = (
        matrix
        .style
        .format(fmt_table_value)
        .apply(
            lambda _: highlight_table(matrix, best_peak, best_by_column),
            axis=None
        )
    )

    st.dataframe(styled)

    show_peak_details(numbers, best_peak)

    show_interval(numbers, best_peak)

    fig_range = plot_input_with_peak_range(numbers, best_peak)
    st.pyplot(fig_range)