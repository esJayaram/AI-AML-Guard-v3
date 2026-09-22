import numpy as np
import pandas as pd

def add_rule_features(df):
    x = df.copy()
    amount = x['Amount'].astype(float)
    q95 = float(amount.quantile(.95)) if len(x) else 0
    q99 = float(amount.quantile(.99)) if len(x) else 0
    rules = pd.Series(0.0, index=x.index)
    reasons = pd.Series('', index=x.index, dtype='string')
    conditions = [
        (amount >= q99, .40, 'Very high transaction amount'),
        (amount >= q95, .20, 'High transaction amount'),
        (x['Sender_account'].eq(x['Receiver_account']), .25, 'Self-transfer pattern'),
        (x['Payment_currency'].ne(x['Received_currency']), .10, 'Currency conversion'),
        (x['Sender_bank_location'].ne(x['Receiver_bank_location']), .10, 'Cross-border bank location'),
    ]
    for cond, score, reason in conditions:
        rules += cond.astype(float) * score
        reasons = reasons.mask(cond & reasons.eq(''), reason)
        reasons = reasons.mask(cond & reasons.ne(''), reasons + '; ' + reason)
    x['rule_score'] = rules.clip(0, 1)
    x['rule_reasons'] = reasons
    return x

def risk_band(score):
    if score >= .85: return 'Critical'
    if score >= .65: return 'High'
    if score >= .35: return 'Medium'
    return 'Low'

def make_explanation(row):
    reasons = str(row.get('rule_reasons','')).strip()
    if reasons and reasons != '<NA>': return reasons
    if row.get('anomaly_score',0) >= .75: return 'Unusual transaction pattern detected by anomaly model'
    return 'Low observed risk pattern'
