# Student Performance Rater 🎓📈

An End-to-End Machine Learning project designed to predict and analyze student performance based on various demographic and educational factors. This repository encompasses the complete machine learning lifecycle, from exploratory data analysis (EDA) and model training to deployment via a web interface.

## 🚀 Project Overview

The Student Performance Rater aims to understand how variables like gender, ethnicity, parental education level, lunch type, and test preparation course influence a student's test scores. By leveraging machine learning, this system provides a predicted score based on these real-world inputs.

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Machine Learning & Data Science:** Scikit-Learn, Pandas, NumPy, CatBoost
* **Web Framework:** Flask / Python web backend
* **Frontend:** HTML/CSS (Jinja Templates)
* **Environment & Packaging:** `setup.py`, `requirements.txt`

## 📁 Project Structure

```text
Student-Performance-Rater/
│
├── artifacts/          # Stored models, preprocessors (e.g., model.pkl, preprocessor.pkl)
├── catboost_info/      # CatBoost model training logs and metadata
├── logs/               # Application and model training execution logs
├── notebook/           # Jupyter notebooks for EDA and Model Training
├── src/                # Modularized source code 
│   ├── components/     # Data ingestion, transformation, and model trainer scripts
│   └── pipeline/       # Training and prediction pipelines
├── templates/          # HTML templates for the web application UI
├── .gitignore          # Files to ignore in version control
├── README.md           # Project documentation
├── requirements.txt    # Required Python dependencies
└── setup.py            # Script for packaging the project as a local module
