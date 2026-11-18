"""Streamlit product review analysis app. Start with: streamlit run app.py"""
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from preprocessing import clean_text
from database import initialize, add_review, history, delete_review
from fake_review import analyze_suspiciousness
from aspect_analysis import analyze_aspects
from reports import csv_bytes, excel_bytes, chart_png_bytes
from prediction import predict_one
from config import settings

DATA_PATH = settings.dataset_path
st.set_page_config(page_title=settings.app_title, page_icon=settings.app_icon, layout=settings.app_layout)
initialize()

@st.cache_resource
def load_bundle():
    path = settings.sentiment_bundle_path
    if path.exists(): return joblib.load(path)
    old_model, old_vec = settings.sentiment_model_path, settings.vectorizer_path
    if old_model.exists() and old_vec.exists():
        return {"model": joblib.load(old_model), "legacy_vectorizer": joblib.load(old_vec), "metrics": {}}
    return None

@st.cache_resource
def load_rating_model():
    path = settings.rating_model_path
    return joblib.load(path) if path.exists() else None

def predict(text, bundle):
    return predict_one(text, bundle)

def find_review_text_column(columns):
    """Find a review text header despite spaces, underscores, or hyphens."""
    aliases = {"review", "review text", "text", "content"}
    for column in columns:
        normalized = " ".join(str(column).strip().lower().replace("_", " ").replace("-", " ").split())
        if normalized in aliases:
            return column
    return None

bundle = load_bundle()
if bundle is None:
    st.error("Sentiment model files are missing. Run `python train_model.py` from this project folder, then restart the app.")
    st.stop()

st.title(f"{settings.app_icon} {settings.app_title}")
st.caption("Explore product feedback, classify a review, or analyze a CSV with the saved machine-learning model.")
st.markdown("""
<style>
    :root {
        --ink: #17243b;
        --muted: #62718a;
        --line: #e1e8f2;
        --surface: #ffffff;
        --surface-soft: #f7f9fd;
        --accent: #4f46e5;
        --accent-dark: #3730a3;
        --accent-soft: #eef2ff;
        --shadow: 0 8px 24px rgba(28, 43, 72, .07);
    }
    html, body { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    .stApp { background: linear-gradient(180deg, #f7f9fd 0%, #f1f5fb 100%); color: var(--ink); }
    [data-testid="stAppViewContainer"] { background: transparent; }
    [data-testid="stHeader"] { background: rgba(247, 249, 253, .88); }
    .block-container { max-width: 1320px; padding: 2rem 2rem 3.5rem; }
    [data-testid="stSidebar"] { background: #f0f3fb; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { background: transparent; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--muted); }
    [data-testid="stSidebar"] label { color: var(--ink) !important; font-weight: 550; }
    h1, h2, h3, h4 { color: var(--ink) !important; letter-spacing: -.025em; line-height: 1.2; }
    h1 { font-weight: 760 !important; margin-bottom: .35rem !important; }
    h2, h3 { font-weight: 700 !important; }
    p, li, label, [data-testid="stMarkdownContainer"] { color: var(--ink); }
    [data-testid="stCaptionContainer"] { color: var(--muted); }
    [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: .5rem; flex-wrap: wrap; border-bottom: 1px solid var(--line); }
    [data-testid="stTabs"] button[role="tab"] { min-height: 3rem; padding: .65rem 1rem; color: var(--muted); font-weight: 650; border-radius: 11px 11px 0 0; }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color: var(--accent-dark); background: var(--accent-soft); }
    [data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--line); border-radius: 16px; padding: 1.1rem 1.2rem; box-shadow: var(--shadow); }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; font-weight: 600; }
    [data-testid="stMetricValue"] { color: var(--ink) !important; font-weight: 750; }
    [data-testid="stMetricDelta"] { font-weight: 600; }
    .stButton > button, .stDownloadButton > button { min-height: 2.75rem; border: 1px solid #d9e1ef; border-radius: 11px; background: var(--surface); color: var(--ink); font-weight: 650; transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease; }
    .stButton > button:hover, .stDownloadButton > button:hover { border-color: #a5b4fc; color: var(--accent-dark); box-shadow: 0 5px 14px rgba(79, 70, 229, .12); transform: translateY(-1px); }
    .stButton > button[kind="primary"] { background: var(--accent); border-color: var(--accent); color: #fff; }
    .stButton > button[kind="primary"]:hover { background: var(--accent-dark); border-color: var(--accent-dark); color: #fff; }
    [data-baseweb="input"] > div, [data-baseweb="textarea"] > div, [data-baseweb="select"] > div { border-color: #cfd8e8; border-radius: 11px; background: var(--surface); }
    [data-baseweb="input"] input, [data-baseweb="textarea"] textarea { color: var(--ink); }
    [data-baseweb="input"] > div:focus-within, [data-baseweb="textarea"] > div:focus-within, [data-baseweb="select"] > div:focus-within { border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(79, 70, 229, .13); }
    [data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid var(--line); border-radius: 13px; overflow: hidden; background: var(--surface); }
    [data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 13px; background: rgba(255,255,255,.8); }
    [data-testid="stFileUploader"] section { border: 1.5px dashed #b9c6dc; border-radius: 14px; background: var(--surface-soft); }
    [data-testid="stFileUploader"] section:hover { border-color: var(--accent); background: var(--accent-soft); }
    [data-testid="stAlert"] { border-radius: 12px; }
    [data-testid="stChatMessage"] { border: 1px solid var(--line); border-radius: 14px; background: var(--surface); padding: .9rem 1rem; }
    [data-testid="stPlotlyChart"], [data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"] { border-radius: 13px; }
    hr { border-color: var(--line); }
    @media (max-width: 700px) {
        .block-container { padding: 1.2rem .85rem 2.5rem; }
        h1 { font-size: 1.75rem !important; }
        h2 { font-size: 1.4rem !important; }
        [data-testid="stMetric"] { padding: .8rem; }
        [data-testid="stTabs"] button[role="tab"] { padding: .55rem .75rem; }
        .stButton > button, .stDownloadButton > button { white-space: normal; height: auto; }
    }
</style>
""", unsafe_allow_html=True)

def read_dataset():
    try: return pd.read_csv(DATA_PATH)
    except Exception: return pd.DataFrame()

dataset = read_dataset()
text_col = find_review_text_column(dataset.columns)
label_col = next((c for c in dataset.columns if str(c).lower() in ("sentiment", "label")), None)
old_reviews = dataset[text_col].dropna().astype(str).tolist() if text_col else []

product_col = next((c for c in dataset.columns if str(c).strip().lower() in ("product_name", "product")), None)
language_col = next((c for c in dataset.columns if str(c).strip().lower() in ("language", "lang")), None)
page = st.sidebar.radio("Navigate", ["Home Dashboard", "Analysis Workspace", "About Project"], label_visibility="visible")

if page == "Home Dashboard":
    st.header("Product review dashboard")
    st.write("A quick view of the labeled dataset used to train the sentiment model.")
    if dataset.empty or not label_col:
        st.info("A labeled dataset is not available at the configured dataset path.")
    else:
        filtered = dataset[dataset[label_col].notna()].copy()
        filter_columns = st.columns(3 if product_col and language_col else 2)
        if product_col:
            product_options = ["All"] + sorted(filtered[product_col].dropna().astype(str).unique().tolist())
            selected_product = filter_columns[0].selectbox("Product", product_options)
            if selected_product != "All":
                filtered = filtered[filtered[product_col].astype(str) == selected_product]
        sentiment_options = ["All"] + sorted(filtered[label_col].dropna().astype(str).unique().tolist())
        sentiment_slot = filter_columns[1] if product_col else filter_columns[0]
        selected_sentiment = sentiment_slot.selectbox("Sentiment", sentiment_options)
        if selected_sentiment != "All":
            filtered = filtered[filtered[label_col].astype(str) == selected_sentiment]
        if language_col:
            language_options = ["All"] + sorted(filtered[language_col].dropna().astype(str).unique().tolist())
            language_slot = filter_columns[2] if product_col else filter_columns[1]
            selected_language = language_slot.selectbox("Language", language_options)
            if selected_language != "All":
                filtered = filtered[filtered[language_col].astype(str) == selected_language]

        labels = filtered[label_col].fillna("").astype(str).str.strip().str.title()
        total = int(len(filtered))
        counts = labels.value_counts()
        metrics = st.columns(4)
        metrics[0].metric("Total reviews", total)
        for column, name, emoji in zip(metrics[1:], ("Positive", "Neutral", "Negative"), ("😊", "😐", "☹️")):
            number = int(counts.get(name, 0))
            percent = number / total * 100 if total else 0
            column.metric(f"{emoji} {name}", number, f"{percent:.1f}%")
        if total:
            sentiment_counts = counts.reindex(["Positive", "Neutral", "Negative"], fill_value=0)
            charts = st.columns(2)
            charts[0].bar_chart(sentiment_counts)
            pie_fig, pie_ax = plt.subplots()
            pie_ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", startangle=90)
            pie_ax.set_title("Sentiment distribution")
            charts[1].pyplot(pie_fig, use_container_width=True)
            plt.close(pie_fig)
            summary = pd.DataFrame({"Sentiment": sentiment_counts.index, "Reviews": sentiment_counts.values})
            summary["Percent"] = (summary["Reviews"] / total * 100).round(1)
            st.dataframe(summary, use_container_width=True, hide_index=True)
            if product_col:
                st.subheader("Product-wise sentiment")
                st.dataframe(filtered.groupby([product_col, label_col]).size().rename("Reviews").reset_index(), use_container_width=True, hide_index=True)
        st.caption("Dataset label counts are shown here; they are not live prediction totals. Language and product filters appear only when those columns exist.")

elif page == "Analysis Workspace":
    chat_tab, upload_tab = st.tabs(["💬 Chatbot", "📊 Upload file analysis"])
    with upload_tab:

        st.header("Upload and analyze a review file")
        st.caption("Upload a CSV with `id`, `name`, and `review text` columns. The review text is analyzed; ID and name are kept in the results.")

        with st.expander("Training dataset snapshot", expanded=False):
            if dataset.empty:
                st.warning(f"Could not load training data at `{DATA_PATH}`.")
            elif label_col:
                counts = dataset[label_col].dropna().astype(str).value_counts().rename_axis("Sentiment").reset_index(name="Reviews")
                total = int(counts.Reviews.sum())
                counts["Percent"] = (counts.Reviews / total * 100).round(1)
                a, b = st.columns(2)
                a.metric("Labeled training reviews", total)
                b.metric("Sentiment categories", len(counts))
                st.dataframe(counts, hide_index=True, use_container_width=True)
                st.bar_chart(counts.set_index("Sentiment")["Reviews"])
                fig, ax = plt.subplots()
                ax.pie(counts["Reviews"], labels=counts["Sentiment"], autopct="%1.1f%%", startangle=90)
                ax.set_title("Training label distribution")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                product_col = next((c for c in dataset.columns if str(c).strip().lower() in ("product_name", "product")), None)
                if product_col and text_col:
                    st.subheader("Training data by product")
                    st.dataframe(dataset.groupby([product_col, label_col]).size().rename("Reviews").reset_index(), hide_index=True, use_container_width=True)
                st.caption("The displayed predictions use the saved trained model; these are the dataset labels used during training.")
            else:
                st.info("No sentiment label column was detected in the configured training dataset.")

        st.subheader("Choose a review CSV")
        uploaded = st.file_uploader("Upload a CSV with id, name, and review text", type=["csv"], key="chat_csv")
        if uploaded and uploaded.size > settings.max_upload_mb * 1024 * 1024:
            st.error(f"File exceeds the configured limit of {settings.max_upload_mb} MB.")
        elif uploaded and st.button("Analyze uploaded CSV", type="primary"):
            try:
                frame = pd.read_csv(uploaded)
                col = find_review_text_column(frame.columns)
                if col is None:
                    st.error("Review text column not found. Include `review text` (also accepts `review_text`, `review`, `text`, or `content`).")
                else:
                    frame = frame.copy()
                    frame[col] = frame[col].fillna("").astype(str)
                    blank_count = int(frame[col].str.strip().eq("").sum())
                    frame = frame[frame[col].str.strip() != ""]
                    rating_model = load_rating_model()
                    known = old_reviews + [r["review_text"] for r in history()]
                    output = []
                    for _, row in frame.iterrows():
                        review = row[col]
                        sentiment, confidence, probabilities = predict(review, bundle)
                        item = {str(k): v for k, v in row.items() if k != col}
                        item.update({"review_text": review, "predicted_sentiment": str(sentiment), "confidence": confidence})
                        item["predicted_rating"] = int(rating_model.predict([clean_text(review)])[0]) if rating_model else None
                        item.update(analyze_suspiciousness(review, known))
                        item["aspect_results"] = "; ".join(f"{a['Aspect']}: {a['Sentiment']}" for a in analyze_aspects(review))
                        item["class_probabilities"] = probabilities
                        output.append(item)
                        add_review(review, str(sentiment), item["predicted_rating"], confidence, item["suspicious_score"], source="bulk")
                        known.append(review)
                    if not output:
                        st.warning("The CSV contained no non-empty reviews.")
                    else:
                        st.session_state["bulk_results"] = output
                        st.session_state["bulk_file_name"] = uploaded.name
                        st.session_state["bulk_blank_count"] = blank_count
            except Exception as exc:
                st.error(f"Could not analyze this CSV: {exc}")

        if st.session_state.get("bulk_results"):
            output = st.session_state["bulk_results"]
            result = pd.DataFrame(output)
            st.subheader(f"File results: {st.session_state.get('bulk_file_name', 'uploaded CSV')}")
            if st.session_state.get("bulk_blank_count", 0):
                st.info(f"Skipped {st.session_state['bulk_blank_count']} blank review row(s).")
            counts = result["predicted_sentiment"].value_counts()
            total = len(result)
            cols = st.columns(4)
            cols[0].metric("Reviews analyzed", total)
            for index, label in enumerate(("Positive", "Neutral", "Negative"), start=1):
                n = int(counts.get(label, 0))
                cols[index].metric(label, f"{n} ({n/total:.0%})")
            if result["confidence"].notna().any():
                st.caption(f"Mean model confidence: {result['confidence'].dropna().mean():.1%}. Confidence is an estimate, not a guarantee.")
            st.dataframe(result.drop(columns=["class_probabilities"], errors="ignore"), use_container_width=True, hide_index=True)
            chart_counts = counts.reindex(["Positive", "Neutral", "Negative"], fill_value=0)
            chart_col, pie_col = st.columns(2)
            chart_col.bar_chart(chart_counts)
            pie_fig, pie_ax = plt.subplots()
            pie_ax.pie(chart_counts, labels=chart_counts.index, autopct="%1.1f%%", startangle=90)
            pie_ax.set_title("Review sentiment share")
            pie_col.pyplot(pie_fig, use_container_width=True)
            plt.close(pie_fig)
            report_col, excel_col, bar_download_col, pie_download_col = st.columns(4)
            report_col.download_button("Download results CSV", csv_bytes(output), "review_analysis.csv", "text/csv")
            try:
                excel_col.download_button("Download results Excel", excel_bytes(output), "review_analysis.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except ImportError:
                excel_col.info("Install openpyxl for Excel export.")

            bar_download_col.download_button("Download bar chart PNG", chart_png_bytes(chart_counts, "bar"), "sentiment_bar_chart.png", "image/png")
            pie_download_col.download_button("Download pie chart PNG", chart_png_bytes(chart_counts, "pie"), "sentiment_pie_chart.png", "image/png")


    with chat_tab:
        st.header("Review sentiment")
        st.caption("Write or paste a review, then analyze it. English and Tamil characters are preserved in preprocessing.")
        def set_review_sample(value):
            st.session_state["review_input"] = value

        def clear_review():
            st.session_state["review_input"] = ""
            st.session_state["chat_messages"] = []
            st.session_state["last_chat_result"] = None

        sample_cols = st.columns(3)
        sample_cols[0].button("😊 Positive example", on_click=set_review_sample, args=("This product is excellent and works perfectly.",), use_container_width=True)
        sample_cols[1].button("😐 Neutral example", on_click=set_review_sample, args=("The product is okay and works as expected.",), use_container_width=True)
        sample_cols[2].button("☹️ Negative example", on_click=set_review_sample, args=("Poor quality, and it stopped working after one day.",), use_container_width=True)

        chat_messages = st.session_state.get("chat_messages", [])
        if st.session_state.get("last_chat_result") and len(chat_messages) >= 2:
            chat_messages = chat_messages[:-2]
        for message in chat_messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and message.get("mood"):
                    st.markdown(f"### {message['mood']['emoji']} {message['mood']['label']}")
                st.markdown(message["content"])

        review_text = st.text_area(
            "Your product review",
            placeholder="Enter your product review here...",
            height=150,
            key="review_input",
            help="Enter an English or Tamil review. Predictions depend on the languages and examples in the training data.",
        )
        analyze_col, clear_col = st.columns([3, 1])
        analyze_clicked = analyze_col.button("Analyze review", type="primary", use_container_width=True)
        clear_col.button("Clear", on_click=clear_review, use_container_width=True)

        if analyze_clicked:
            if not review_text.strip():
                st.warning("Please enter a review before analyzing it.")
            else:
                with st.spinner("Analyzing your review..."):
                    try:
                        sentiment, confidence, probabilities = predict(review_text, bundle)
                        mood_map = {
                            "Positive": {"emoji": "😊", "label": "Good"},
                            "Neutral": {"emoji": "😐", "label": "Neutral"},
                            "Negative": {"emoji": "☹️", "label": "Bad"},
                        }
                        mood = mood_map.get(str(sentiment), {"emoji": "🔎", "label": str(sentiment)})
                        suspicious = analyze_suspiciousness(review_text, old_reviews + [r["review_text"] for r in history()])
                        aspects = analyze_aspects(review_text)
                        rating_model = load_rating_model()
                        rating = int(rating_model.predict([clean_text(review_text)])[0]) if rating_model else None
                        add_review(review_text, str(sentiment), rating, confidence, suspicious["suspicious_score"])
                        detail = []
                        if confidence is not None:
                            detail.append(f"Model confidence estimate: **{confidence:.1%}**.")
                        if rating is not None:
                            detail.append(f"Estimated rating: **{rating}/5 stars**. This is an estimate.")
                        detail.append(f"Suspiciousness indicator: **{suspicious['suspicious_score']}/100** — {suspicious['suspicious_warning']}. This is not a fake-review verdict.")
                        aspect_summary = ", ".join(f"{item['Aspect']}: {item['Sentiment']}" for item in aspects) if aspects else "No supported aspect was detected."
                        detail.append(f"Aspect analysis: {aspect_summary}")
                        reply = "\n\n".join(detail)
                        st.session_state["last_chat_result"] = {"mood": mood, "content": reply}
                        st.session_state.setdefault("chat_messages", []).extend([
                            {"role": "user", "content": review_text},
                            {"role": "assistant", "content": reply, "mood": mood},
                        ])
                    except Exception as exc:
                        st.error(f"Could not analyze this review: {exc}")

        result = st.session_state.get("last_chat_result")
        if result:
            st.markdown("---")
            st.subheader("Prediction")
            st.markdown(f"## {result['mood']['emoji']} {result['mood']['label']}")
            st.markdown(result["content"])

    with upload_tab:
        with st.expander("Recent review history", expanded=False):
            filter_col, from_col, to_col = st.columns(3)
            history_sentiment = filter_col.selectbox("Sentiment filter", ["All", "Positive", "Negative", "Neutral"], key="history_sentiment")
            history_start = from_col.date_input("From date", value=None, key="history_start")
            history_end = to_col.date_input("To date", value=None, key="history_end")
            history_search = st.text_input("Search saved reviews", key="history_search")
            recent = history(history_sentiment, history_start, history_end, history_search)
            if recent:
                st.dataframe(pd.DataFrame(recent[:100]), use_container_width=True, hide_index=True)
                delete_id = st.selectbox("Delete history record", [row["id"] for row in recent], key="history_delete_id")
                delete_col, csv_col, excel_col = st.columns(3)
                if delete_col.button("Delete selected record", key="history_delete_button"):
                    delete_review(delete_id)
                    st.rerun()
                csv_col.download_button("Download history CSV", csv_bytes(recent), "review_history.csv", "text/csv")
                try:
                    excel_col.download_button("Download history Excel", excel_bytes(recent), "review_history.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                except ImportError:
                    excel_col.caption("Install openpyxl for Excel downloads.")
            else:
                st.caption("Analyzed reviews will appear here.")

        with st.expander("Model evaluation", expanded=False):
            metrics = bundle.get("metrics", {})
            if metrics:
                test = metrics.get("test", {})
                st.write(f"Selected model: **{metrics.get('selected_model')}**")
                st.json(metrics.get("validation_scores", {}))
                a, b, c, d = st.columns(4)
                for column, key in zip((a, b, c, d), ("accuracy", "precision_weighted", "recall_weighted", "f1_weighted")):
                    column.metric(key.replace("_", " ").title(), f"{test.get(key, 0):.3f}")
                st.dataframe(pd.DataFrame(test.get("report", {})).T, use_container_width=True)
            else:
                st.info("Run `python train_model.py` to create model comparison and evaluation results.")

else:
    st.header("About this project")
    st.write("This project analyzes product-review text and estimates whether the sentiment is positive, neutral, or negative.")
    st.subheader("How it works")
    st.markdown(
        "1. Review text is normalized while preserving negation and Tamil characters.\n"
        "2. A saved TF-IDF and calibrated classifier predicts sentiment.\n"
        "3. CSV uploads are analyzed row by row and can be exported with summary charts."
    )
    st.subheader("Built with")
    st.write("Python · Streamlit · Pandas · Scikit-learn · TF-IDF · SQLite · Matplotlib · Joblib")
    st.info("Sentiment, confidence, rating, and suspiciousness outputs are estimates or heuristic indicators. Ratings need genuine rating labels to train, and Tamil/Thanglish quality depends on language coverage in the training set.")
