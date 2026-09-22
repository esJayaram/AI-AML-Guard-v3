import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score

def anomaly_scores(df):
    features = df[['Amount']].copy()
    features['hour'] = df['datetime'].dt.hour.fillna(0)
    features['currency_change'] = (df['Payment_currency'] != df['Received_currency']).astype(int)
    features['cross_border'] = (df['Sender_bank_location'] != df['Receiver_bank_location']).astype(int)
    features = features.replace([np.inf,-np.inf],np.nan).fillna(0)
    if len(features) < 20: return np.zeros(len(features))
    sample_n = min(len(features), 100_000)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(features), sample_n, replace=False)
    model = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=-1)
    model.fit(features.iloc[idx])
    raw = -model.score_samples(features)
    lo, hi = np.percentile(raw, [1,99])
    return np.clip((raw-lo)/(hi-lo+1e-9),0,1)

def supervised_model(df):
    pos = df[df.Is_laundering.eq(1)]
    neg = df[df.Is_laundering.eq(0)]
    if len(pos) < 5 or len(neg) < 10: return {}, pd.DataFrame()
    rng = np.random.default_rng(42)
    pos = pos.sample(min(len(pos), 5000), random_state=42)
    neg = neg.sample(min(len(neg), max(len(pos)*4, 10000)), random_state=42)
    data = pd.concat([pos,neg]).sample(frac=1, random_state=42)
    features = ['Amount','Payment_currency','Received_currency','Sender_bank_location','Receiver_bank_location','Payment_type']
    X, y = data[features], data.Is_laundering
    num = ['Amount']; cat = [c for c in features if c != 'Amount']
    prep = ColumnTransformer([('num', Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)])
    pipe = Pipeline([('prep',prep),('model',LogisticRegression(max_iter=500, class_weight='balanced'))])
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
    pipe.fit(Xtr,ytr); proba=pipe.predict_proba(Xte)[:,1]; pred=(proba>=.5).astype(int)
    metrics={'Precision':precision_score(yte,pred,zero_division=0),'Recall':recall_score(yte,pred,zero_division=0),'F1':f1_score(yte,pred,zero_division=0),'PR-AUC':average_precision_score(yte,proba),'ROC-AUC':roc_auc_score(yte,proba)}
    out=pd.DataFrame({'actual':yte.to_numpy(),'predicted':pred,'probability':proba})
    return metrics,out
