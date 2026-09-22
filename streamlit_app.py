import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from app.data_loader import load_saml_sample, DATA_COLUMNS
from app.detection import add_rule_features, make_explanation, risk_band
from app.ml import anomaly_scores, supervised_model
from app.network import build_network

st.set_page_config(page_title='AI-AML Guard v2', page_icon='🛡️', layout='wide')

st.markdown('''
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:12px;}
[data-testid="stMetricLabel"] * {color:#475569 !important;}
[data-testid="stMetricValue"] * {color:#0b1f3a !important;font-weight:700 !important;}
</style>
''', unsafe_allow_html=True)

st.title('🛡️ AI-AML Guard v2')
st.caption('SAML-D powered AML transaction monitoring, risk scoring, analytics and network investigation')

with st.sidebar:
    st.header('Data Controls')
    rows = st.slider('Rows loaded', 10_000, 500_000, 250_000, 10_000)
    default_path = Path('data/raw/SAML-D.csv')
    path_text = st.text_input('SAML-D CSV path', str(default_path))
    dataset_path = Path(path_text)
    st.caption('For the large SAML-D file, the app reads only the selected number of rows for responsive analysis.')
    if dataset_path.exists():
        st.success('SAML-D detected')
        st.caption(f'Dataset: {dataset_path.resolve()}')
    else:
        st.error('SAML-D.csv not found')
        st.info('Place SAML-D.csv in data/raw/ or enter its full path above.')
        st.stop()

@st.cache_data(show_spinner='Loading SAML-D sample...')
def cached_load(path, nrows):
    return load_saml_sample(path, nrows)

try:
    df = cached_load(str(dataset_path), rows)
except Exception as exc:
    st.error('Dataset loading failed.')
    st.exception(exc)
    st.stop()

if df.empty:
    st.error('No rows were loaded from SAML-D.')
    st.stop()

# Core derived fields
try:
    df = add_rule_features(df)
    df['anomaly_score'] = anomaly_scores(df)
    df['risk_score'] = (0.55 * df['anomaly_score'] + 0.45 * df['rule_score']).clip(0, 1)
    df['risk_level'] = df['risk_score'].map(risk_band)
    df['explanation'] = df.apply(make_explanation, axis=1)
except Exception as exc:
    st.error('Risk scoring failed.')
    st.exception(exc)
    st.stop()

# KPIs
transactions = len(df)
value = float(pd.to_numeric(df['Amount'], errors='coerce').fillna(0).sum())
suspicious = int(pd.to_numeric(df['Is_laundering'], errors='coerce').fillna(0).sum())
susp_pct = suspicious / transactions * 100 if transactions else 0
high_critical = int(df['risk_level'].isin(['High', 'Critical']).sum())

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric('Transactions', f'{transactions:,}')
m2.metric('Value', f'{value:,.0f}')
m3.metric('Suspicious', f'{suspicious:,}')
m4.metric('Suspicious %', f'{susp_pct:.2f}%')
m5.metric('High/Critical', f'{high_critical:,}')

# Tabs
tab_overview, tab_alerts, tab_model, tab_explorer, tab_network, tab_quality = st.tabs([
    'Overview', 'AML Alerts', 'Model', 'Explorer', 'Network', 'Quality'
])

with tab_overview:
    st.subheader('Overview')
    status_df = df['Is_laundering'].map({0:'Normal', 1:'Suspicious'}).fillna('Unknown').value_counts().rename_axis('status').reset_index(name='count')
    payment_df = df.groupby('Payment_type', dropna=False, as_index=False)['Amount'].sum().sort_values('Amount', ascending=False)
    daily = (df.assign(date=pd.to_datetime(df['datetime'], errors='coerce').dt.normalize())
               .dropna(subset=['date'])
               .groupby('date', as_index=False)
               .agg(transaction_value=('Amount','sum'), transactions=('Amount','size'), suspicious=('Is_laundering','sum'))
               .sort_values('date'))

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.pie(status_df, names='status', values='count', hole=.55, title='Normal vs Suspicious Transactions'), width='stretch')
    with c2:
        st.plotly_chart(px.bar(payment_df, x='Payment_type', y='Amount', title='Transaction Value by Payment Type'), width='stretch')

    if not daily.empty:
        st.plotly_chart(px.line(daily, x='date', y='transaction_value', markers=True, render_mode='svg', title='Daily Transaction Value'), width='stretch')
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(px.line(daily, x='date', y='transactions', markers=True, render_mode='svg', title='Daily Transaction Count'), width='stretch')
        with c4:
            st.plotly_chart(px.bar(daily, x='date', y='suspicious', title='Suspicious Transactions by Date'), width='stretch')

with tab_alerts:
    st.subheader('AML Alerts')
    a1, a2, a3 = st.columns(3)
    a1.metric('Critical', int((df.risk_level == 'Critical').sum()))
    a2.metric('High', int((df.risk_level == 'High').sum()))
    a3.metric('Rule-triggered', int((df.rule_score > 0).sum()))
    st.plotly_chart(px.histogram(df, x='risk_score', nbins=40, title='Risk Score Distribution'), width='stretch')
    alert_cols = [c for c in ['Date','Time','Sender_account','Receiver_account','Amount','Payment_currency','Received_currency','Payment_type','risk_score','risk_level','explanation','Is_laundering'] if c in df.columns]
    alerts = df.sort_values(['risk_score','Amount'], ascending=False)[alert_cols].head(500)
    st.dataframe(alerts, width='stretch', hide_index=True)
    st.download_button('Download Top 500 Alerts CSV', alerts.to_csv(index=False).encode('utf-8'), 'aml_alerts.csv', 'text/csv')

with tab_model:
    st.subheader('Supervised Model Performance')
    if df['Is_laundering'].nunique() < 2 or df['Is_laundering'].sum() < 5:
        st.warning('Not enough positive labels in the loaded sample to train the supervised model.')
    else:
        with st.spinner('Training model on a balanced sample...'):
            metrics, pred_df = supervised_model(df)
        cols = st.columns(5)
        for col, (name, val) in zip(cols, metrics.items()):
            col.metric(name.replace('_',' ').upper(), 'N/A' if pd.isna(val) else f'{val:.3f}')
        if pred_df is not None and not pred_df.empty:
            cm = pd.crosstab(pred_df['actual'], pred_df['predicted'], rownames=['Actual'], colnames=['Predicted'], dropna=False)
            st.dataframe(cm, width='stretch')

with tab_explorer:
    st.subheader('Transaction Explorer')
    e1, e2, e3 = st.columns(3)
    levels = e1.multiselect('Risk level', ['Critical','High','Medium','Low'], default=['Critical','High','Medium','Low'])
    payments = sorted(df['Payment_type'].dropna().astype(str).unique().tolist())
    pay = e2.multiselect('Payment type', payments, default=payments)
    truth = e3.selectbox('Ground truth', ['All','Suspicious','Normal'])
    mask = df['risk_level'].isin(levels) & df['Payment_type'].astype(str).isin(pay)
    if truth == 'Suspicious': mask &= df['Is_laundering'].eq(1)
    elif truth == 'Normal': mask &= df['Is_laundering'].eq(0)
    show_cols = [c for c in ['Date','Time','Sender_account','Receiver_account','Amount','Payment_currency','Received_currency','Payment_type','Is_laundering','risk_score','risk_level','explanation'] if c in df.columns]
    result = df.loc[mask, show_cols].sort_values('risk_score', ascending=False).head(1000)
    st.caption(f'{len(result):,} transactions shown (maximum 1,000).')
    st.dataframe(result, width='stretch', hide_index=True)
    st.download_button('Download Filtered Transactions CSV', result.to_csv(index=False).encode('utf-8'), 'filtered_transactions.csv', 'text/csv')

with tab_network:
    st.subheader('Transaction Network Analysis')
    graph, edges, nodes = build_network(df)
    if edges.empty:
        st.info('No network edges available for the selected sample.')
    else:
        n1, n2, n3 = st.columns(3)
        n1.metric('Nodes', graph.number_of_nodes())
        n2.metric('Edges', graph.number_of_edges())
        n3.metric('High-risk edges', int((edges['risk_score'] >= .75).sum()))
        st.plotly_chart(px.scatter(edges.head(100), x='amount', y='risk_score', size='transactions', hover_data=['Sender_account','Receiver_account'], title='Top Network Edges'), width='stretch')
        st.dataframe(nodes.head(100), width='stretch', hide_index=True)

with tab_quality:
    st.subheader('Data Quality')
    quality = pd.DataFrame({
        'column': df.columns,
        'dtype': [str(df[c].dtype) for c in df.columns],
        'missing': [int(df[c].isna().sum()) for c in df.columns],
        'unique': [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(quality, width='stretch', hide_index=True)
    q1, q2, q3 = st.columns(3)
    q1.metric('Duplicate rows', f'{int(df.duplicated().sum()):,}')
    q2.metric('Columns', len(df.columns))
    q3.metric('Missing cells', f'{int(df.isna().sum().sum()):,}')

st.caption('AI-AML Guard v2 • SAML-D synthetic AML dataset • Use for research, demonstration and portfolio purposes.')
