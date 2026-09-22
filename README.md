# AI-AML Guard v3

Intelligent Anti-Money Laundering and Transaction Monitoring System built around the SAML-D synthetic AML transaction dataset.

## Features
- Real SAML-D CSV ingestion
- Responsive row sampling up to 500,000 transactions
- AML rule-based risk scoring
- Isolation Forest anomaly detection
- Supervised Logistic Regression evaluation
- Risk bands: Low, Medium, High, Critical
- Alert explanations
- Interactive Plotly dashboards
- Transaction explorer and CSV export
- Transaction network analysis with NetworkX
- Data-quality checks

## Dataset
Download SAML-D separately and place the CSV at: https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml

`data/raw/SAML-D.csv`

Do not commit the large dataset to GitHub. Check the dataset's current license and Kaggle terms before redistribution.

## Windows setup

```powershell
cd AI-AML-Guard-v3
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place your dataset here:

data\raw\SAML-D.csv`

Run:

```powershell
streamlit run streamlit_app.py
```

## If your CSV is elsewhere
Enter its full path in the sidebar, for example:

`D:\AI-AML-Guard-v3\data\raw\SAML-D.csv`

## Recommended starting point
Use 100,000–250,000 rows for a normal laptop. Increase to 500,000 if memory and CPU are sufficient.

## Project structure

```text
AI-AML-Guard-v3/
├── streamlit_app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── app/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── detection.py
│   ├── ml.py
│   └── network.py
└── data/
    └── raw/
        └── SAML-D.csv   # user-provided; not included in repository
```

## Important
This project is for research, demonstration and portfolio use. It is not a production AML compliance system and should not be used as the sole basis for financial-crime decisions.


