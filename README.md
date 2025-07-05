DSP PROJECT
# 🚀 Water Quality Data Ingestion & Monitoring Pipeline

### *An End-to-End Data Engineering and Machine Learning Project*

---

## 📚 Table of Contents

1. [Project Overview](#project-overview)
2. [Project Architecture](#project-architecture)
3. [Tech Stack](#tech-stack)
4. [Folder Structure](#folder-structure)
5. [Step-by-Step Setup](#step-by-step-setup)
6. [How the Pipeline Works](#how-the-pipeline-works)
7. [Airflow Pipeline Details](#airflow-pipeline-details)
8. [Database Structure](#database-structure)
9. [API Endpoints](#api-endpoints)
10. [Streamlit Dashboard](#streamlit-dashboard)
11. [Microsoft Teams Alerts](#microsoft-teams-alerts)
12. [Future Improvements](#future-improvements)

---

## 📖 Project Overview

This project demonstrates a **real-time, production-ready pipeline** that ingests, validates, processes, and monitors **water quality datasets**.

The pipeline includes:

* Automated data ingestion using **Apache Airflow**
* Data validation and separation into **good** and **bad** datasets
* Saving processing statistics and bad data to **PostgreSQL**
* Sending alerts to **Microsoft Teams** for bad data detection
* Generating **HTML reports** for each ingestion run
* Real-time interaction with the data via a **FastAPI service**
* Data visualization using a **Streamlit dashboard**

---

## 🏗️ Project Architecture

```text
┌───────────┐       ┌──────────┐       ┌──────────┐
│  Raw Data │──────▶│  Airflow │──────▶│ PostgreSQL│
└───────────┘       └────┬─────┘       └──────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Good / Bad Data │
                  │   File Storage  │
                  └─────────────────┘
                           │
                           ▼
               ┌────────────────────┐
               │ HTML Reports &     │
               │ Teams Notifications│
               └────────────────────┘
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
        ┌──────────────┐       ┌──────────────┐
        │   Streamlit  │       │    FastAPI   │
        │   Dashboard  │       │  ML Services │
        └──────────────┘       └──────────────┘
```

---

## 🛠️ Tech Stack

* **Apache Airflow**: Orchestration and scheduling
* **Docker Compose**: Container orchestration
* **PostgreSQL**: Metadata storage
* **Redis**: Celery message broker
* **FastAPI**: Real-time API for model serving
* **Streamlit**: Interactive dashboard
* **Python**: Data validation, ETL, ML Model
* **Microsoft Teams Webhook**: Alerts for bad data

---

## 📂 Folder Structure

```text
ml_pipeline_project/
├── dags/                          # Airflow DAG files
│   └── data_ingestion_pipeline.py # Airflow DAG
├── Scripts/                       # Python scripts
│   ├── data_ingestion.py          # Ingestion and validation logic
│   ├── data_splitter.py           # Chunking raw data
│   ├── ingestion_job.py           # Standalone ingestion job
│   └── Clean_bad_data.py          # Cleaning bad data script
├── data/                          # Data folders
│   ├── ingestion/raw_data_chunks/ # Raw data chunks
│   ├── good_data/                 # Validated good data
│   └── bad_data/                  # Invalid bad data
├── logs/                          # Airflow logs
├── plugins/                       # Airflow plugins (if needed)
├── reports/                       # Auto-generated HTML reports
├── streamlit_app.py               # Streamlit dashboard
├── Fast_API.py                    # FastAPI server
├── Ml_model.py                    # Machine Learning model
├── docker-compose.yaml            # Airflow, PostgreSQL, Redis containers
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## 🚀 Step-by-Step Setup

### 1. Clone Repository

```bash
git clone <repo-url>
cd ml_pipeline_project
```

### 2. Setup Python Environment

```bash
conda create -n myenv python=3.11
conda activate myenv
pip install -r requirements.txt
```

### 3. Start Airflow & Dependencies

```bash
docker compose up
```

> Airflow Webserver: `http://localhost:8085`
> Default Login: `airflow / airflow`

---

## ⚙️ How the Pipeline Works

1. **Data Ingestion**: Reads CSV chunks from the raw data folder.
2. **Data Validation**: Separates good and bad data using custom validation rules.
3. **Data Storage**:

   * Good and bad data are saved in separate folders.
   * Statistics and bad data are saved in PostgreSQL.
4. **HTML Reporting**: Auto-generates an HTML validation report per chunk.
5. **Teams Alerts**: Sends real-time alerts if bad data is detected.
6. **Streamlit Dashboard**: Visualizes the ingestion process and data statistics.
7. **FastAPI**: Exposes endpoints to interact with the model and data in real-time.

---

## 🛠️ Airflow Pipeline Details

### DAG: `data_ingestion_pipeline`

* **Task:** Executes the `data_ingestion.py` script inside the Airflow container.
* **Schedule:** Manual trigger
* **Command:**

```python
bash_command='python /opt/airflow/scripts/data_ingestion.py'
```

---

## 🗄️ PostgreSQL Tables

1. **ingestion\_statistics**

   * filename: Name of the processed chunk
   * total\_rows: Total rows in the file
   * good\_rows: Number of good rows
   * bad\_rows: Number of bad rows
   * ingestion\_time: Timestamp of ingestion

2. **bad\_data\_table**

   * Stores all bad rows with source file information

---

## ⚡ API Endpoints (FastAPI)

```bash
python Fast_API.py
```

* `GET /` : Health check
* `POST /predict` : Sends good data to the model and returns predictions

---

## 📊 Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

* Visualizes good vs bad data counts
* Displays ingestion trends
* Shows real-time ingestion status

---

## 💬 Microsoft Teams Alerts

* **Criticality Levels:**

  * 🚨 High: > 5000 bad rows
  * ⚠️ Medium: 1000 - 5000 bad rows
  * ℹ️ Low: < 1000 bad rows

> Make sure to **replace the webhook URL** in the `data_ingestion.py` with your own Teams webhook.

---

## 🔮 Future Improvements

* Integrate email alerts
* Automate chunking via cron jobs
* Schedule retraining of the ML model
* Add more data validation rules with Great Expectations
* Dockerize FastAPI and Streamlit services

