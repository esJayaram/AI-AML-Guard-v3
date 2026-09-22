import pandas as pd
import networkx as nx

def build_network(df):
    cols=['Sender_account','Receiver_account','Amount','risk_score']
    work=df[cols].copy().sort_values(['risk_score','Amount'],ascending=False).head(12000)
    if work.empty: return nx.DiGraph(), pd.DataFrame(), pd.DataFrame()
    g=nx.DiGraph()
    edge_rows=[]
    grouped=work.groupby(['Sender_account','Receiver_account'],as_index=False).agg(amount=('Amount','sum'),transactions=('Amount','size'),risk_score=('risk_score','max'))
    for r in grouped.itertuples(index=False):
        g.add_edge(str(r.Sender_account),str(r.Receiver_account),amount=float(r.amount),transactions=int(r.transactions),risk_score=float(r.risk_score))
    nodes=[]
    for node, deg in g.degree(): nodes.append({'account':node,'degree':deg,'in_degree':g.in_degree(node),'out_degree':g.out_degree(node)})
    return g, grouped.sort_values(['risk_score','amount'],ascending=False), pd.DataFrame(nodes).sort_values('degree',ascending=False)
