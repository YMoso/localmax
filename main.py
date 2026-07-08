import numpy as np
import pandas as pd
import streamlit as st


def fmt_num(x):
    if float(x).is_integer():
        return str(int(x))
    return str(round(float(x), 4))


def make_matrix(numbers):
    n = len(numbers)
    arr = np.array(numbers, dtype=float)
    result = {}

    for size in range(1, n + 1):
        col_values = []

        for i in range(n - size + 1):
            total = arr[i:i + size].sum()
            squared_total = total ** 2
            col_values.append(round(squared_total, 2))

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


st.set_page_config(page_title="LocalMax", layout="wide")

default_df = pd.DataFrame({
    "number": [1.0, 2.0, 1.25, 2.5, 2.75, 3.0, 4.5, 2.2, 1.1, 2.4]
})

edited_df = st.data_editor(
    default_df,
    num_rows="dynamic",
    hide_index=False,
    column_config={
        "number": st.column_config.NumberColumn(
            "Number",
            help="Enter one numeric value per row",
            required=False
        )
    },
    key="numbers"
)

numbers = edited_df["number"].dropna().astype(float).tolist()

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

    show_interval(numbers, best_peak)

if best_peak is None:
    st.warning("No local peak was found.")
    st.dataframe(matrix)
else:
    styled = (matrix.style.format("{:.3f}").apply(lambda _: highlight_table(matrix, best_peak, best_by_column),axis=None))

    st.dataframe(styled)