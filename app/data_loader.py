from pathlib import Path
import pandas as pd

DATA_COLUMNS = ['Time','Date','Sender_account','Receiver_account','Amount','Payment_currency','Received_currency','Sender_bank_location','Receiver_bank_location','Payment_type','Is_laundering','Laundering_type']

def _parse_datetime(df):
    date = df['Date'].astype(str).str.strip()
    time = df['Time'].astype(str).str.strip()
    combined = pd.to_datetime(date + ' ' + time, errors='coerce')
    if combined.isna().mean() > 0.5:
        combined = pd.to_datetime(date, errors='coerce')
    return combined

def load_saml_sample(path: str, nrows: int) -> pd.DataFrame:
    path = Path(path)
    if not path.exists(): raise FileNotFoundError(f'Dataset not found: {path}')
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    missing = [c for c in DATA_COLUMNS if c not in df.columns]
    if missing: raise ValueError(f'SAML-D columns missing: {missing}')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0.0)
    df['Is_laundering'] = pd.to_numeric(df['Is_laundering'], errors='coerce').fillna(0).astype(int)
    df['datetime'] = _parse_datetime(df)
    if df['datetime'].isna().all():
        raise ValueError('Could not parse Date + Time into datetime. Check the SAML-D file format.')
    for c in ['Sender_account','Receiver_account','Payment_currency','Received_currency','Sender_bank_location','Receiver_bank_location','Payment_type','Laundering_type']:
        df[c] = df[c].astype('string').fillna('Unknown')
    return df
