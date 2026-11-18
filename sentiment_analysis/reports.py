"""Export analysis records to CSV and Excel."""
from io import BytesIO, StringIO
import pandas as pd
import matplotlib.pyplot as plt

def csv_bytes(records):
    return pd.DataFrame(records).to_csv(index=False).encode("utf-8-sig")

def excel_bytes(records):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        pd.DataFrame(records).to_excel(writer, index=False, sheet_name="Reviews")
    return out.getvalue()

def chart_png_bytes(counts, chart_type="bar"):
    """Render a sentiment count Series as a downloadable PNG chart."""
    figure, axis = plt.subplots(figsize=(7, 4))
    if chart_type == "bar":
        axis.bar(counts.index, counts.values, color=["#55a868", "#6889c4", "#d95f5f"][:len(counts)])
        axis.set_ylabel("Reviews")
        axis.set_title("Review sentiment counts")
    elif chart_type == "pie":
        axis.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
        axis.set_title("Review sentiment share")
    else:
        plt.close(figure)
        raise ValueError("chart_type must be 'bar' or 'pie'.")
    output = BytesIO()
    figure.savefig(output, format="png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output.getvalue()

