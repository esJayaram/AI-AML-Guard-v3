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
├── .gitignore<img width="1539" height="756" alt="Screenshot 2026-09-22 184933" src="https://github.com/user-attachments/assets/7174cf97-3c0d-49bc-a0f1-7dc9d13740b2" />
<img width="1537" height="905" alt="Screenshot 2026-09-22 184951" src="https://github.com/user-attachments/assets/b0c8126a-9bc8-4037-9b99-97f852725364" />
<img width="1521" height="861" alt="Screenshot 2026-09-22 185009" src="https://github.com/user-attachments/assets/146c9f61-78a6-4c53-ab28-f6406582a707" />
<img width="1507" height="721" alt="Screenshot 2026-09-22 185032" src="https://github.com/user-attachments/assets/6cf26bc5-3532-4b44-8b11-336ffc73e6d1" />

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


