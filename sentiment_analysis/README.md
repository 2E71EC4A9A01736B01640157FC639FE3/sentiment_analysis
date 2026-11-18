# Product Review Sentiment Studio

A Streamlit application for exploring product review sentiment, classifying individual reviews, and analyzing uploaded review files with a trained machine learning model.

## Features

- Dashboard with labeled dataset sentiment counts and filters.
- Chat style review analyzer with positive, neutral, or negative results.
- CSV batch analysis with per-review predictions, confidence estimates, aspect summaries, and suspiciousness indicators.
- Downloadable result CSV and Excel files, plus sentiment distribution charts as PNG files.
- Local review history stored in SQLite.
- Optional estimated star ratings when a separately trained rating model is available.

## Requirements

- Python 3.10 or newer
- Packages listed in `requirements.txt`

## Install and run

From this project directory, create and activate a virtual environment, install the packages, and start Streamlit:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The repository includes a trained sentiment model and the example training dataset, so the app can run without training a new model first.

## Configure the app

Copy `.env.example` to `.env` to override the default settings. For example:

```powershell
Copy-Item .env.example .env
```

Available settings include the app title, dataset/model paths, model training parameters, history database path, and upload size limit. `.env` is a local file and is excluded from Git.

## Analyze a CSV file

In **Analysis Workspace → Upload file analysis**, upload a CSV containing a review text column. The recommended format is:

```csv
id,name,review text
101,Wireless Headphones,"Excellent sound quality and comfortable to wear."
102,Travel Mug,"It works as expected, but nothing stands out."
103,Desk Lamp,"Stopped working after two days. Very poor quality."
```

The `id` and `name` columns are retained in the result. The app accepts `review text`, `review_text`, `review`, `text`, or `content` as the review column header. Empty review rows are skipped. Upload size is controlled by `MAX_UPLOAD_MB` (20 MB by default).

After analysis, the app displays sentiment totals, a results table, a bar chart, and a pie chart. Use the download buttons to save the result CSV/Excel files and chart PNGs.

## Train the sentiment model

To retrain the model, provide a CSV with a review text column and a sentiment label column, then set `DATASET_PATH` in `.env` if needed. Accepted review headers include `review`, `review_text`, `text`, and `content`; accepted label headers include `sentiment`, `label`, and `target`.

```powershell
python train_model.py
```

Training writes the model bundle and evaluation chart files to `MODEL_DIR` (the project directory by default). The example `dataset.csv` contains `review` and `sentiment` columns.

## Notes

- Predictions and confidence values are estimates and depend on the training data.
- Aspect analysis only reports supported aspects found in the review text.
- The suspiciousness score is a heuristic warning, not a determination that a review is fake.
- The rating model is optional and requires real numeric rating labels to train.
- Review history is local to the application and stored in `review_history.db`.
