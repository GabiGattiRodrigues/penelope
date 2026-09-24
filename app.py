"""
Penélope — quem vai continuar fiel? Previsão de renovação de um programa de fidelidade pago,
do M0 ao M11, com GAM. App Streamlit (PT/EN).
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import joblib

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))
from variaveis import FEATURES, NUM, CAT  # noqa: E402
import gam  # noqa: E402,F401  (necessário para carregar os modelos)

ART = RAIZ / "artefatos"
DATA = RAIZ / "data"

st.set_page_config(page_title="Penélope · renovação de fidelidade", page_icon=str(RAIZ / "assets" / "penelope.svg"), layout="wide")

# ------------------------------------------------------------------ cores
AZUL_ESC, AZUL, CEU, AMBAR = "#1E3A8A", "#2563EB", "#0EA5E9", "#D97706"
COR_AMOSTRA = {"treino": AZUL, "teste": CEU, "out-of-time": AMBAR}
INK, INK2, GRID = "#0F172A", "#475569", "#E2E8F0"
BLUES = [[0, "#EFF6FF"], [0.25, "#BFDBFE"], [0.5, "#60A5FA"], [0.75, "#2563EB"], [1, "#1E3A8A"]]
COR_RISCO = {"Alto risco": "#B91C1C", "Médio risco": "#D97706", "Baixo risco": "#15803D"}

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
.kpi {border:1px solid #E2E8F0; border-radius:12px; padding:14px 16px; background:#F8FAFC;}
.kpi .v {font-size:1.7rem; font-weight:700; color:#0F172A; line-height:1.2;}
.kpi .l {font-size:.82rem; color:#475569;}
.caixa {border-left:4px solid #2563EB; background:#EFF6FF; padding:12px 16px; border-radius:6px; margin:6px 0 14px;}
.passo {border:1px solid #BFDBFE; border-radius:12px; padding:14px; background:#F8FAFC; height:100%;}
.passo h4 {margin:0 0 6px; color:#1E3A8A;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ idioma
with st.sidebar:
    _q = str(st.query_params.get("lang", "pt")).lower()
    lang = st.radio("Idioma / Language", ["PT", "EN"], index=1 if _q == "en" else 0,
                    horizontal=True, label_visibility="collapsed")
    st.image(str(RAIZ / "assets" / "penelope.png"), width="stretch")


def L(pt, en):
    return pt if lang == "PT" else en


NOMES = {
    "pedidos_acum": L("Pedidos acumulados", "Cumulative orders"),
    "atrasometro": L("Atrasômetro", "Delay-meter"),
    "ticket_medio": L("Ticket médio (R$)", "Average ticket (R$)"),
    "economia_acum": L("Economia acumulada no programa (R$)", "Cumulative program savings (R$)"),
    "qtd_setores": L("Qtd. de setores comprados", "Number of store sections bought"),
    "uso_continuo": L("Comprou medicamento de uso contínuo", "Bought continuous-use medication"),
    "sem_ritmo": L("Sem ritmo definido (< 2 pedidos)", "No rhythm yet (< 2 orders)"),
    "canal": L("Canal de adesão", "Sign-up channel"),
    "pagamento": L("Forma de pagamento", "Payment method"),
    "setor_principal": L("Setor principal", "Main section"),
}
TRAD_VAL = {
    "Cartão recorrente": "Recurring card", "Pix/boleto": "Pix/bank slip", "Loja": "Store",
    "Medicamentos": "Medicines", "Dermocosméticos": "Dermocosmetics", "Higiene e beleza": "Hygiene & beauty",
    "Infantil": "Baby & kids", "Suplementos": "Supplements", "Conveniência": "Convenience",
    "Sem compra": "No purchase", "Não": "No", "Sim": "Yes",
    "Alto risco": "High risk", "Médio risco": "Medium risk", "Baixo risco": "Low risk",
    "treino": "train", "teste": "test", "Sem ritmo (< 2 pedidos)": "No rhythm (< 2 orders)", "out-of-time": "out-of-time",
    "desenvolvimento": "development", "carteira ativa": "active base",
}
ACOES_EN = {
    "Ativação: 1ª compra com cupom do programa": "Activation: 1st purchase with program coupon",
    "Resgate: lembrete de recompra + oferta no setor principal": "Win-back: repurchase reminder + offer in main section",
    "Mostrar economia: extrato do que o programa já devolveu": "Show savings: statement of what the program gave back",
    "Lembrete de recompra no ritmo do cliente": "Repurchase reminder at the customer's own pace",
    "Cross-sell: apresentar um setor novo": "Cross-sell: introduce a new section",
    "Manter: régua padrão + convite para renovação antecipada": "Keep: standard journey + early-renewal invite",
}


def tv(x):
    x = str(x)
    if lang == "PT":
        return x
    return TRAD_VAL.get(x, ACOES_EN.get(x, x.replace("R$", "R$").replace("Sem compra", "No purchase")))


def num(v, casas=1):
    """Formata número no padrão do idioma (PT: 1.234,5 · EN: 1,234.5)."""
    t = f"{v:,.{casas}f}"
    return t.replace(",", "X").replace(".", ",").replace("X", ".") if lang == "PT" else t


def layout(fig, h=380, **kw):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white",
                      paper_bgcolor="white", font=dict(color=INK, size=13),
                      legend=dict(orientation="h", y=1.08, x=0), hoverlabel=dict(bgcolor="white"), **kw)
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="")
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def kpi(col, valor, rotulo):
    col.markdown(f'<div class="kpi"><div class="v">{valor}</div><div class="l">{rotulo}</div></div>',
                 unsafe_allow_html=True)


# ------------------------------------------------------------------ dados
@st.cache_data
def carregar():
    met = json.loads((ART / "metricas.json").read_text())
    biv = pd.read_parquet(ART / "bivariada.parquet")
    cart = pd.read_parquet(ART / "carteira_escorada.parquet")
    membros = pd.read_parquet(DATA / "membros.parquet")
    hist = pd.read_parquet(ART / "scores_historico.parquet")
    return met, biv, cart, membros, hist


@st.cache_resource
def modelos():
    return joblib.load(ART / "modelos_gam.joblib")


met, biv, cart, membros, hist = carregar()
MESES = [f"M{m}" for m in range(12)]

with st.sidebar:
    st.markdown(f"### Penélope")
    st.caption(L("Quem vai continuar fiel? Previsão de renovação de um programa de fidelidade pago, "
                 "do 1º ao 11º mês de vida do membro.",
                 "Who will stay loyal? Renewal prediction for a paid loyalty program, "
                 "from the member's 1st to 11th month."))
    st.markdown(L("**Stack:** Python · scikit-learn (GAM com splines) · Plotly · Streamlit",
                  "**Stack:** Python · scikit-learn (spline GAM) · Plotly · Streamlit"))
    st.markdown(L("Reconstrução pública, com dados sintéticos, de um projeto real que fiz no varejo.",
                  "Public rebuild, with synthetic data, of a real project I did in retail."))
    st.markdown("[Portfólio · Gabi Gatti](https://gabigattirodrigues.github.io/)")

st.title(L("Penélope · quem vai continuar fiel?", "Penélope · who will stay loyal?"))
st.caption(L("Como Penélope tecendo à espera de Ulisses: o modelo acompanha a fidelidade do membro mês a mês, "
             "do M0 ao M11, e antecipa quem renova no M12.",
             "Like Penelope weaving while waiting for Odysseus: the model follows the member's loyalty month by "
             "month, M0 to M11, and anticipates who renews at M12."))

abas = st.tabs([L("O caso", "The case"), L("Análise exploratória", "Exploratory analysis"), L("Variáveis × tempo", "Variables × time"),
                L("Performance", "Performance"), L("O que o modelo aprendeu", "What the model learned"),
                L("Aplicação: carteira ativa", "Application: active base"), L("Simulador", "Simulator"),
                L("Metodologia e glossário", "Methodology & glossary")])

# ================================================================== 1. O caso
with abas[0]:
    st.markdown(L(
        """**Cenário.** Uma rede de drogarias tem o **Vida+**, um programa de fidelidade **pago e anual**
(R$ 119,90/ano) com frete grátis e 5% de cashback. No fim dos 12 meses, o membro decide se renova.
A pergunta do negócio: **dá pra saber antes quem não vai renovar, e agir a tempo?**""",
        """**Scenario.** A drugstore chain runs **Vida+**, a **paid annual** loyalty program (R$ 119.90/year)
with free shipping and 5% cashback. After 12 months the member decides whether to renew.
The business question: **can we tell in advance who won't renew, and act in time?**"""))

    a = met["meses"]
    c = st.columns(4)
    kpi(c[0], f"{len(membros):,}".replace(",", "."), L("membros sintéticos (32 safras)", "synthetic members (32 cohorts)"))
    kpi(c[1], "12", L("modelos GAM, um por mês de vida (M0–M11)", "GAM models, one per month of tenure (M0–M11)"))
    kpi(c[2], f"{a['0']['amostras']['out-of-time']['auc']:.2f} → {a['11']['amostras']['out-of-time']['auc']:.2f}",
        L("AUC out-of-time, do M0 ao M11", "Out-of-time AUC, from M0 to M11"))
    kpi(c[3], f"{a['11']['amostras']['out-of-time']['ks']*100:.0f}",
        L("KS out-of-time no M11", "Out-of-time KS at M11"))

    st.markdown("#### " + L("Como o caso foi montado", "How the case was built"))
    p = st.columns(3)
    dv = {k: f"{v:,}".replace(",", ".") for k, v in membros["recorte"].value_counts().items()}
    passos = [
        (L("1 · Desenvolvimento", "1 · Development"), L("safras jan–dez/2024", "cohorts Jan–Dec/2024"),
         L(f"{dv['desenvolvimento']} membros. 70% treino, 30% teste, separados por membro.",
           f"{dv['desenvolvimento']} members. 70% train, 30% test, split by member.")),
        (L("2 · Out-of-time", "2 · Out-of-time"), L("safras jan–ago/2025", "cohorts Jan–Aug/2025"),
         L(f"{dv['out-of-time']} membros que o modelo nunca viu, de um período posterior. É a prova de que ele se sustenta.",
           f"{dv['out-of-time']} members the model never saw, from a later period. Proof it holds up.")),
        (L("3 · Aplicação", "3 · Application"), L("safras set/2025–ago/2026", "cohorts Sep/2025–Aug/2026"),
         L(f"{dv['carteira ativa']} membros ativos, ainda sem resposta. Recebem score, faixa de risco e ação sugerida.",
           f"{dv['carteira ativa']} active members, outcome unknown. They get a score, risk band and suggested action.")),
    ]
    for col, (t, s, d) in zip(p, passos):
        col.markdown(f'<div class="passo"><h4>{t}</h4><b>{s}</b><br>{d}</div>',
                     unsafe_allow_html=True)

    st.markdown("#### " + L("As duas variáveis que mais conversaram com o negócio", "The two variables the business cared about most"))
    st.markdown(L(
        """- **Pedidos acumulados**: quanto o membro já usou o programa até aquele mês.
- **Atrasômetro**: quanto o membro está **fora do próprio ritmo de compra**. Se ele compra a cada 30 dias e está no 45º dia
sem comprar, o atrasômetro marca **45 ÷ 30 − 1 = +50%**. Ele substitui a recência "pura": 20 dias sem comprar é normal
para quem compra a cada 40 dias, e é alerta para quem compra a cada 10.""",
        """- **Cumulative orders**: how much the member has used the program up to that month.
- **Delay-meter**: how far the member is **off their own buying rhythm**. If they buy every 30 days and are on day 45
without buying, the delay-meter reads **45 ÷ 30 − 1 = +50%**. It replaces plain recency: 20 days without buying is
normal for someone who buys every 40 days, and a warning for someone who buys every 10. A rhythm only exists from
the 2nd order on; before that the member is flagged as "no rhythm"."""))

    with st.expander(L("Do case original × o que é acréscimo desta versão", "Original case × what this version adds")):
        st.markdown(L(
            """**Veio do projeto real (varejo, programa de assinatura):** modelo GAM; previsão da renovação do M12 a partir de
cada mês de vida, do M0 ao M11; pedidos acumulados e atrasômetro como variáveis-chave; quantidade de setores
comprados e flag de categoria-chave (lá, um tipo de produto; aqui, medicamento de uso contínuo); leitura bivariada
das variáveis × mês de vida × probabilidade, em mapa de calor, para os stakeholders.

**Acréscimos desta versão pública:** cenário de drogaria e base 100% sintética; validação out-of-time com PSI;
lista de ação da carteira ativa com ação sugerida; simulador de membro.""",
            """**From the real project (retail, subscription program):** GAM model; M12 renewal predicted from each month of
tenure, M0 to M11; cumulative orders and delay-meter as key variables; number of sections bought and a key-category
flag (there, a product type; here, continuous-use medication); bivariate reading of variables × month × probability
as a heatmap for stakeholders.

**Added in this public version:** drugstore scenario and 100% synthetic data; out-of-time validation with PSI;
active-base action list with suggested action; member simulator."""))

# ================================================================== 2. Dados
with abas[1]:
    hist_m = membros[membros["recorte"] != "carteira ativa"]
    t = hist_m.groupby(["safra", "recorte"])["renovou"].agg(["mean", "size"]).reset_index()
    fig = go.Figure()
    for r, cor in [("desenvolvimento", AZUL), ("out-of-time", AMBAR)]:
        d = t[t["recorte"] == r]
        fig.add_bar(x=d["safra"], y=d["mean"], name=tv(r), marker_color=cor,
                    customdata=d["size"], hovertemplate="%{x}<br>%{y:.1%} · n=%{customdata}<extra></extra>")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_layout(bargap=0.15, title=L("Taxa de renovação por safra de adesão", "Renewal rate by sign-up cohort"))
    st.plotly_chart(layout(fig, 360), width="stretch")
    st.caption(L("A safra de 2025 renova um pouco menos: mudança real de comportamento que o out-of-time precisa aguentar.",
                 "2025 cohorts renew a bit less: a real behavior shift the out-of-time test has to survive."))

    st.divider()
    st.markdown("#### " + L("Explorador de variáveis", "Variable explorer"))
    st.caption(L("Base de desenvolvimento (treino + teste). Cada foto é o retrato do membro no fim daquele mês de vida.",
                 "Development base (train + test). Each snapshot is the member at the end of that month of tenure."))
    dev = hist[hist["amostra"] != "out-of-time"].copy()
    dev["grupo"] = np.where(dev["renovou"] == 1, L("Renovou", "Renewed"), L("Não renovou", "Did not renew"))
    COR_GRUPO = {L("Renovou", "Renewed"): AZUL, L("Não renovou", "Did not renew"): AMBAR}
    e1, e2 = st.columns([1, 2])
    var_e = e1.selectbox(L("Variável", "Variable"), FEATURES, format_func=lambda v: NOMES[v], key="eda_var")
    fotos = e2.multiselect(L("Fotos (meses de vida)", "Snapshots (months of tenure)"), MESES,
                           default=["M0", "M2", "M4", "M6", "M8", "M11"], key="eda_m")
    fotos = sorted(fotos, key=lambda x: int(x[1:])) or ["M6"]
    mf = [int(x[1:]) for x in fotos]
    de = dev[dev["mes_vida"].isin(mf)]

    if var_e in NUM:
        base_e = de[de["sem_ritmo"] == 0] if var_e == "atrasometro" else de
        teto = float(dev[var_e].quantile(0.99))
        fig = go.Figure()
        for gname, cor in COR_GRUPO.items():
            g = base_e[base_e["grupo"] == gname]
            fig.add_box(x=("M" + g["mes_vida"].astype(str)), y=g[var_e].clip(upper=teto), name=gname,
                        marker_color=cor, boxpoints=False, line=dict(width=1.5))
        fig.update_layout(boxmode="group", boxgap=0.35,
                          title=L(f"Distribuição de {NOMES[var_e]} por foto", f"Distribution of {NOMES[var_e]} by snapshot"))
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=fotos)
        if var_e == "atrasometro":
            fig.update_yaxes(tickformat="+.0%")
        elif var_e in ("pedidos_acum", "qtd_setores"):
            fig.update_yaxes(dtick=2 if var_e == "pedidos_acum" else 1, tickformat="d")
        st.plotly_chart(layout(fig, 400), width="stretch")
        if var_e == "atrasometro":
            st.caption(L("Só membros com ritmo definido (2+ pedidos). Os demais aparecem como \"% sem ritmo\" na tabela.",
                         "Only members with a defined rhythm (2+ orders). The rest show up as \"% no rhythm\" in the table."))

        g = dev.groupby(["mes_vida", "grupo"])
        med = (dev[dev["sem_ritmo"] == 0] if var_e == "atrasometro" else dev).groupby(["mes_vida", "grupo"])[var_e].median().unstack()
        fig = go.Figure()
        for gname, cor in COR_GRUPO.items():
            fig.add_scatter(x=MESES, y=med[gname], name=gname, mode="lines+markers",
                            line=dict(color=cor, width=2.5), marker=dict(size=8))
        if var_e == "atrasometro":
            fig.update_yaxes(tickformat="+.0%")
        fig.update_layout(title=L(f"Mediana de {NOMES[var_e]}, M0 a M11", f"Median {NOMES[var_e]}, M0 to M11"))
        st.plotly_chart(layout(fig, 320), width="stretch")

        linhas = []
        for m in mf:
            for gname in COR_GRUPO:
                x = dev[(dev["mes_vida"] == m) & (dev["grupo"] == gname)]
                xv = x.loc[x["sem_ritmo"] == 0, var_e] if var_e == "atrasometro" else x[var_e]
                r = {L("Foto", "Snapshot"): f"M{m}", L("Grupo", "Group"): gname, "n": len(x),
                     L("Média", "Mean"): xv.mean(), "P25": xv.quantile(.25), L("Mediana", "Median"): xv.median(),
                     "P75": xv.quantile(.75)}
                if var_e == "atrasometro":
                    r[L("% sem ritmo", "% no rhythm")] = x["sem_ritmo"].mean()
                else:
                    r[L("% zero", "% zero")] = (x[var_e] == 0).mean()
                linhas.append(r)
        tab = pd.DataFrame(linhas)
        pct_col = tab.columns[-1]
        if var_e == "atrasometro":
            fnum = lambda v: f"{v:+.0%}"
        elif var_e in ("pedidos_acum", "qtd_setores"):
            fnum = lambda v: num(v, 1) if v != int(v) else num(v, 0)
        else:
            fnum = lambda v: num(v, 1)
        st.dataframe(tab.style.format({c: fnum for c in tab.columns[3:-1]} |
                                      {pct_col: "{:.0%}", "n": lambda v: num(v, 0)}),
                     width="stretch", hide_index=True)
    else:
        niveis = sorted(dev[var_e].unique())
        tons_m = ["#BFDBFE", "#93C5FD", "#60A5FA", "#3B82F6", "#2563EB", "#1D4ED8", "#1E40AF", "#1E3A8A"]
        fig = go.Figure()
        for i, m in enumerate(mf):
            d = dev[dev["mes_vida"] == m][var_e].value_counts(normalize=True).reindex(niveis).fillna(0)
            fig.add_bar(x=[tv(n) for n in d.index], y=d.values, name=f"M{m}",
                        marker_color=tons_m[min(int(i * 7 / max(len(mf) - 1, 1)), 7)],
                        hovertemplate=f"M{m} · %{{x}}: %{{y:.1%}}<extra></extra>")
        fig.update_yaxes(tickformat=".0%")
        fig.update_xaxes(type="category")
        fig.update_layout(barmode="group", title=L(f"Composição da base por {NOMES[var_e]}", f"Base mix by {NOMES[var_e]}"))
        st.plotly_chart(layout(fig, 380), width="stretch")
        st.caption(L("Tom mais escuro = foto mais avançada.", "Darker shade = later snapshot."))
        tab = (de.groupby(["mes_vida", var_e]).agg(n=("renovou", "size"), renov=("renovou", "mean")).reset_index())
        tab["share"] = tab["n"] / tab.groupby("mes_vida")["n"].transform("sum")
        tab = pd.DataFrame({L("Foto", "Snapshot"): "M" + tab["mes_vida"].astype(str),
                            NOMES[var_e]: tab[var_e].map(tv), "n": tab["n"],
                            L("% da base", "% of base"): tab["share"], L("Renovação", "Renewal"): tab["renov"]})
        st.dataframe(tab.style.format({"n": lambda v: num(v, 0), L("% da base", "% of base"): "{:.1%}", L("Renovação", "Renewal"): "{:.1%}"}),
                     width="stretch", hide_index=True, height=300)

    st.markdown("#### " + L("Correlação entre as variáveis numéricas", "Correlation between numeric variables"))
    mc = st.select_slider(L("Foto", "Snapshot"), MESES, value="M6", key="corr_m")
    dc = dev[dev["mes_vida"] == int(mc[1:])]
    cols_c = NUM + ["uso_continuo"]
    cor = dc[cols_c].corr(method="spearman")
    DIV = [[0, AMBAR], [0.5, "#F1F5F9"], [1, AZUL_ESC]]
    fig = go.Figure(go.Heatmap(z=cor.values, x=[NOMES[c] for c in cols_c], y=[NOMES[c] for c in cols_c],
                               zmin=-1, zmax=1, colorscale=DIV, xgap=2, ygap=2,
                               text=np.round(cor.values, 2), texttemplate="%{text}",
                               hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>", colorbar=dict(thickness=12)))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(title=L(f"Correlação de Spearman · {mc}", f"Spearman correlation · {mc}"))
    st.plotly_chart(layout(fig, 440), width="stretch")
    st.caption(L("Pedidos, economia e setores andam juntos (quem compra mais usa mais o benefício e circula por mais setores). "
                 "O GAM lida bem com isso por ser penalizado, mas é por isso que o efeito de cada uma deve ser lido em conjunto.",
                 "Orders, savings and sections move together (heavier buyers use the benefit more and shop more sections). "
                 "The penalized GAM handles it, but it is why each effect should be read together with the others."))

    with st.expander(L("Amostra da base de variáveis (foto M6)", "Sample of the feature table (M6 snapshot)")):
        st.dataframe(hist[hist["mes_vida"] == 6].head(200), width="stretch", hide_index=True)

# ================================================================== 3. Variáveis × tempo
with abas[2]:
    st.markdown(L(
        "Leitura **bivariada** que ia para os stakeholders: como cada variável se comporta ao longo do mês de vida e "
        "qual a chance de renovação em cada faixa. Base: safras de desenvolvimento.",
        "The **bivariate** view shown to stakeholders: how each variable behaves over the months of tenure and the "
        "renewal chance in each band. Base: development cohorts."))
    c1, c2 = st.columns([2, 1])
    var = c1.selectbox(L("Variável", "Variable"), FEATURES, format_func=lambda v: NOMES[v],
                       index=FEATURES.index("atrasometro"))
    medida = c2.radio(L("Cor do mapa", "Heatmap color"),
                      [L("Taxa de renovação real", "Actual renewal rate"), L("Probabilidade média do modelo", "Model's average probability")])
    col_val = "taxa" if medida.startswith(("Taxa", "Actual")) else "prob"

    d = biv[(biv["variavel"] == var) & (biv["n"] >= 30)].copy()
    d["faixa_t"] = d["faixa"].map(tv)
    ordem = d.drop_duplicates("faixa").sort_values("ordem")["faixa_t"].tolist()
    z = d.pivot(index="faixa_t", columns="mes_vida", values=col_val).reindex(ordem)
    n = d.pivot(index="faixa_t", columns="mes_vida", values="n").reindex(ordem)
    fig = go.Figure(go.Heatmap(
        z=z.values, x=[f"M{c}" for c in z.columns], y=z.index, colorscale=BLUES, zmin=float(np.floor(np.nanmin(z.values) * 10) / 10), zmax=float(np.ceil(np.nanmax(z.values) * 10) / 10),
        text=np.where(np.isnan(z.values), "", np.vectorize(lambda v: f"{v:.0%}")(np.nan_to_num(z.values))),
        texttemplate="%{text}", textfont=dict(size=11), customdata=n.values, xgap=2, ygap=2,
        colorbar=dict(title="", tickformat=".0%", thickness=12),
        hovertemplate=L("%{y} · %{x}<br>renovação: %{z:.1%}<br>n = %{customdata:,}<extra></extra>",
                        "%{y} · %{x}<br>renewal: %{z:.1%}<br>n = %{customdata:,}<extra></extra>")))
    fig.update_yaxes(title=NOMES[var], type="category", categoryorder="array",
                     categoryarray=ordem if var in NUM else ordem[::-1], showgrid=False)
    fig.update_xaxes(title=L("mês de vida", "month of tenure"))
    fig.update_layout(title=L(f"Mapa de calor: {NOMES[var]} × mês de vida", f"Heatmap: {NOMES[var]} × month of tenure"))
    st.plotly_chart(layout(fig, 460), width="stretch")
    st.caption(L("Células com menos de 30 membros ficam em branco.", "Cells with fewer than 30 members are left blank."))

    st.markdown("#### " + L("Curva por faixa, mês a mês", "Curve by band, month by month"))
    sel = st.multiselect(L("Meses para comparar", "Months to compare"), MESES, default=["M0", "M3", "M6", "M11"])
    tons = ["#BFDBFE", "#93C5FD", "#60A5FA", "#3B82F6", "#2563EB", "#1D4ED8", "#1E40AF", "#1E3A8A"]
    sel = sorted(sel, key=lambda s: int(s[1:]))
    fig = go.Figure()
    for i, mm in enumerate(sel):
        dm = d[d["mes_vida"] == int(mm[1:])].sort_values("ordem")
        cor = tons[min(int(i * (len(tons) - 1) / max(len(sel) - 1, 1)), len(tons) - 1)]
        fig.add_scatter(x=dm["faixa_t"], y=dm[col_val], name=mm, mode="lines+markers",
                        line=dict(color=cor, width=2.5), marker=dict(size=8),
                        customdata=dm["n"], hovertemplate=f"{mm} · %{{x}}<br>%{{y:.1%}} · n=%{{customdata:,}}<extra></extra>")
    fig.update_yaxes(tickformat=".0%", title=L("renovação", "renewal"))
    fig.update_xaxes(title=NOMES[var], type="category", categoryorder="array", categoryarray=ordem)
    st.plotly_chart(layout(fig, 380), width="stretch")
    st.caption(L("Tom mais escuro = mês mais avançado.", "Darker shade = later month."))

# ================================================================== 4. Performance
with abas[3]:
    rows = []
    for m in range(12):
        for amo, v in met["meses"][str(m)]["amostras"].items():
            rows.append(dict(mes=f"M{m}", amostra=amo, auc=v["auc"], ks=v["ks"], gini=v["gini"], n=v["n"]))
    perf = pd.DataFrame(rows)

    c1, c2 = st.columns(2)
    for col, metr, tit in [(c1, "auc", "AUC"), (c2, "ks", "KS")]:
        fig = go.Figure()
        for amo, cor in COR_AMOSTRA.items():
            d = perf[perf["amostra"] == amo]
            fig.add_scatter(x=d["mes"], y=d[metr], name=tv(amo), mode="lines+markers",
                            line=dict(color=cor, width=2.5, dash="dot" if amo == "out-of-time" else "solid"),
                            marker=dict(size=8), hovertemplate=f"%{{x}} · {tv(amo)}: %{{y:.3f}}<extra></extra>")
        if metr == "ks":
            for y0, y1, cor_z, rot in [(0, .2, "#FEE2E2", L("fraco", "weak")), (.2, .3, "#FEF3C7", L("aceitável", "acceptable")),
                                       (.3, .4, "#DBEAFE", L("bom", "good")), (.4, 1, "#BFDBFE", L("muito bom", "very good"))]:
                fig.add_hrect(y0=y0, y1=y1, fillcolor=cor_z, opacity=0.45, line_width=0, layer="below",
                              annotation_text=rot, annotation_position="top left",
                              annotation_font=dict(size=11, color=INK2))
            fig.update_yaxes(range=[0.1, 0.7], tickformat=".0%")
        fig.update_layout(title=L(f"{tit} por mês de vida", f"{tit} by month of tenure"))
        col.plotly_chart(layout(fig, 340), width="stretch")
    st.markdown(L(
        "Treino, teste e out-of-time andam juntos: **sem overfitting** e **estável no tempo**. O poder cresce do M0 ao "
        "M11 porque o membro vai revelando o comportamento dele. Nos primeiros meses o KS é baixo (quase ninguém tem ritmo "
        "de compra ainda); **a partir do M4–M5 ele passa de 35–40, na faixa boa**, que é quando dá tempo de agir antes "
        "da renovação. Régua de KS usada: < 20 fraco · 20–30 aceitável · 30–40 bom · ≥ 40 muito bom.",
        "Train, test and out-of-time move together: **no overfitting** and **stable over time**. Power grows from M0 "
        "to M11 as the member reveals their behavior."))

    st.divider()
    mes = st.select_slider(L("Mês de vida para detalhar", "Month to drill into"), MESES, value="M6")
    mm = met["meses"][mes[1:]]
    cc = st.columns(4)
    for col, amo in zip(cc[:3], ["treino", "teste", "out-of-time"]):
        v = mm["amostras"][amo]
        kpi(col, f"{v['auc']:.3f}", f"AUC {tv(amo)} · KS {v['ks']*100:.1f} · Gini {v['gini']*100:.1f}")
    kpi(cc[3], f"{mm['psi_oot']:.3f}", L("PSI treino × out-of-time (< 0,10 = estável)", "PSI train × out-of-time (< 0.10 = stable)"))

    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(color="#94A3B8", dash="dash", width=1), showlegend=False, hoverinfo="skip")
    for amo, cor in COR_AMOSTRA.items():
        r = mm["amostras"][amo]["roc"]
        fig.add_scatter(x=r["fpr"], y=r["tpr"], name=tv(amo), line=dict(color=cor, width=2.5))
    fig.update_xaxes(title=L("falsos positivos", "false positive rate"), range=[0, 1])
    fig.update_yaxes(title=L("verdadeiros positivos", "true positive rate"), range=[0, 1])
    fig.update_layout(title=L(f"Curva ROC · {mes}", f"ROC curve · {mes}"))
    c1.plotly_chart(layout(fig, 380), width="stretch")

    k = mm["amostras"]["out-of-time"]["curva_ks"]
    gap = np.array(k["renovou"]) - np.array(k["nao_renovou"])
    i = int(np.argmax(gap))
    fig = go.Figure()
    fig.add_scatter(x=k["pop"], y=k["renovou"], name=L("renovou", "renewed"), line=dict(color=AZUL, width=2.5))
    fig.add_scatter(x=k["pop"], y=k["nao_renovou"], name=L("não renovou", "did not renew"), line=dict(color=AMBAR, width=2.5))
    fig.add_shape(type="line", x0=k["pop"][i], x1=k["pop"][i], y0=k["nao_renovou"][i], y1=k["renovou"][i],
                  line=dict(color=INK, width=2))
    fig.add_annotation(x=k["pop"][i], y=(k["renovou"][i] + k["nao_renovou"][i]) / 2, text=f"KS = {gap[i]*100:.1f}",
                       showarrow=False, xanchor="left", xshift=6, font=dict(color=INK))
    fig.update_xaxes(title=L("% da base, do maior para o menor score", "% of base, highest to lowest score"), tickformat=".0%")
    fig.update_yaxes(title=L("% acumulado", "cumulative %"), tickformat=".0%")
    fig.update_layout(title=L(f"Curva KS · out-of-time · {mes}", f"KS curve · out-of-time · {mes}"))
    c2.plotly_chart(layout(fig, 380), width="stretch")

    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(color="#94A3B8", dash="dash", width=1), showlegend=False, hoverinfo="skip")
    for amo in ["teste", "out-of-time"]:
        cal = mm["amostras"][amo]["calibracao"]
        fig.add_scatter(x=cal["prevista"], y=cal["observada"], name=tv(amo), mode="lines+markers",
                        line=dict(color=COR_AMOSTRA[amo], width=2.5), marker=dict(size=8))
    fig.update_xaxes(title=L("probabilidade prevista", "predicted probability"), tickformat=".0%")
    fig.update_yaxes(title=L("renovação observada", "observed renewal"), tickformat=".0%")
    fig.update_layout(title=L("Calibração (por decil)", "Calibration (by decile)"))
    c1.plotly_chart(layout(fig, 380), width="stretch")

    psi = pd.DataFrame({"mes": MESES, "psi": [met["meses"][str(m)]["psi_oot"] for m in range(12)]})
    fig = go.Figure(go.Bar(x=psi["mes"], y=psi["psi"], marker_color=AZUL, hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0.1, line=dict(color=AMBAR, dash="dash"), annotation_text=L("0,10: atenção", "0.10: watch"),
                  annotation_position="top left")
    fig.update_yaxes(range=[0, 0.12])
    fig.update_layout(title=L("PSI do score: treino × out-of-time", "Score PSI: train × out-of-time"), bargap=0.3)
    c2.plotly_chart(layout(fig, 380), width="stretch")

    st.markdown("#### " + L(f"Tabela por decil · out-of-time · {mes}", f"Decile table · out-of-time · {mes}"))
    tdec = pd.DataFrame(mm["amostras"]["out-of-time"]["decis"])
    tdec = tdec[["decil", "n", "prob_media", "taxa_renovacao", "pct_renov_acum", "pct_nao_renov_acum", "ks"]]
    tdec.columns = [L("Decil (1 = mais provável renovar)", "Decile (1 = most likely to renew)"), "n",
                    L("Prob. média", "Avg. prob."), L("Renovação real", "Actual renewal"),
                    L("% renovou acum.", "Cum. % renewed"), L("% não renovou acum.", "Cum. % not renewed"), "KS"]
    st.dataframe(tdec.style.format({c: "{:.1%}" for c in tdec.columns[2:]}), width="stretch", hide_index=True)
    st.caption(L("Decis definidos pelos cortes do treino (como seria em produção).",
                 "Deciles use the training cut-offs (as it would be in production)."))

# ================================================================== 5. O que o modelo aprendeu
with abas[4]:
    st.markdown(L(
        "No GAM, cada variável entra como uma **curva própria** e as curvas se somam. Então dá pra ver o efeito de cada "
        "uma isolado: acima de zero, **aumenta** a chance de renovar; abaixo, **diminui** (escala do logito).",
        "In a GAM each variable enters as **its own curve** and the curves add up, so each effect can be read on its "
        "own: above zero it **raises** the renewal odds; below, it **lowers** them (logit scale)."))

    imp = pd.DataFrame({m: met["meses"][str(m)]["importancia"] for m in range(12)})
    imp = imp.loc[imp.mean(axis=1).sort_values(ascending=False).index]
    fig = go.Figure(go.Heatmap(
        z=imp.values, x=MESES, y=[NOMES[v] for v in imp.index], colorscale=BLUES, xgap=2, ygap=2,
        colorbar=dict(thickness=12), hovertemplate="%{y} · %{x}<br>%{z:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(title=L("Peso de cada variável por mês de vida (desvio da contribuição no logito)",
                              "Weight of each variable by month (std. of its logit contribution)"))
    st.plotly_chart(layout(fig, 400), width="stretch")
    st.markdown(L(
        "No começo pesam o **perfil** (forma de pagamento, uso contínuo, amplitude de setores). Conforme o ano passa, "
        "o **comportamento** assume: o atrasômetro vira a variável nº 1 e pedidos/economia ganham força.",
        "Early on, **profile** matters (payment method, continuous-use, section breadth). As the year goes by, "
        "**behavior** takes over: the delay-meter becomes #1 and orders/savings gain weight."))

    c1, c2 = st.columns([2, 1])
    var2 = c1.selectbox(L("Curva da variável", "Variable curve"), FEATURES, format_func=lambda v: NOMES[v],
                        index=FEATURES.index("atrasometro"), key="forma")
    sel2 = c2.multiselect(L("Meses", "Months"), MESES, default=["M2", "M6", "M11"], key="forma_m")
    sel2 = sorted(sel2, key=lambda s: int(s[1:]))
    tons = ["#93C5FD", "#3B82F6", "#1E3A8A", "#60A5FA", "#2563EB", "#1D4ED8"]
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="#94A3B8", width=1))
    for i, mm_ in enumerate(sel2):
        f = met["meses"][mm_[1:]]["formas"][var2]
        cor = ["#93C5FD", "#60A5FA", "#3B82F6", "#2563EB", "#1D4ED8", "#1E3A8A"][
            min(int(i * 5 / max(len(sel2) - 1, 1)), 5)]
        x = [tv(v) for v in f["x"]] if var2 not in NUM else f["x"]
        if var2 in NUM:
            fig.add_scatter(x=x, y=f["y"], name=mm_, line=dict(color=cor, width=2.5))
        else:
            fig.add_bar(x=x, y=f["y"], name=mm_, marker_color=cor)
    if var2 == "atrasometro":
        fig.update_xaxes(tickformat="+.0%")
    fig.update_xaxes(title=NOMES[var2])
    fig.update_yaxes(title=L("efeito no logito da renovação", "effect on renewal logit"))
    fig.update_layout(title=L(f"Efeito de {NOMES[var2]}", f"Effect of {NOMES[var2]}"), barmode="group")
    st.plotly_chart(layout(fig, 400), width="stretch")

# ================================================================== 6. Carteira ativa
with abas[5]:
    st.markdown(L(
        "O modelo aplicado na **carteira ativa** (adesões de set/2025 a ago/2026, foto em 31/08/2026). Cada membro é "
        "escorado pelo GAM do mês de vida em que está hoje.",
        "The model applied to the **active base** (sign-ups Sep/2025–Aug/2026, snapshot on 2026-08-31). Each member is "
        "scored by the GAM for their current month of tenure."))
    cc = st.columns(4)
    kpi(cc[0], f"{len(cart):,}".replace(",", "."), L("membros ativos", "active members"))
    kpi(cc[1], f"{cart['prob'].mean():.0%}", L("renovação esperada", "expected renewal"))
    for col, fx in zip(cc[2:], ["Alto risco", "Médio risco"]):
        q = (cart["faixa_risco"] == fx).sum()
        kpi(col, f"{q:,}".replace(",", "."), f"{tv(fx)} ({q/len(cart):.0%})")
    st.caption(L("Faixas: alto risco < 40% de chance de renovar · médio 40–70% · baixo ≥ 70%.",
                 "Bands: high risk < 40% renewal chance · medium 40–70% · low ≥ 70%."))

    c1, c2 = st.columns(2)
    g = cart.groupby(["mes_vida", "faixa_risco"]).size().unstack(fill_value=0)
    fig = go.Figure()
    for fx in ["Alto risco", "Médio risco", "Baixo risco"]:
        fig.add_bar(x=[f"M{i}" for i in g.index], y=g[fx], name=tv(fx), marker_color=COR_RISCO[fx],
                    hovertemplate=f"%{{x}} · {tv(fx)}: %{{y}}<extra></extra>")
    fig.update_layout(barmode="stack", bargap=0.2, title=L("Membros por mês de vida e faixa de risco",
                                                           "Members by month and risk band"))
    c1.plotly_chart(layout(fig, 360), width="stretch")

    a_ = cart[cart["faixa_risco"] != "Baixo risco"]["acao_sugerida"].value_counts().sort_values()
    fig = go.Figure(go.Bar(x=a_.values, y=[tv(i) for i in a_.index], orientation="h", marker_color=AZUL,
                           hovertemplate="%{y}: %{x}<extra></extra>"))
    fig.update_layout(title=L("Ações sugeridas (alto e médio risco)", "Suggested actions (high & medium risk)"))
    c2.plotly_chart(layout(fig, 360), width="stretch")

    st.markdown("#### " + L("Lista de ação", "Action list"))
    f1, f2, f3 = st.columns(3)
    fx_sel = f1.multiselect(L("Faixa de risco", "Risk band"), ["Alto risco", "Médio risco", "Baixo risco"],
                            default=["Alto risco"], format_func=tv)
    prazo = f2.slider(L("Renova em até (dias)", "Renews within (days)"), 0, 365, 120)
    setor = f3.multiselect(L("Setor principal", "Main section"), sorted(cart["setor_principal"].unique()), format_func=tv)
    lst = cart[cart["faixa_risco"].isin(fx_sel) & (cart["dias_para_renovar"] <= prazo)]
    if setor:
        lst = lst[lst["setor_principal"].isin(setor)]
    lst = lst.sort_values("prob")
    view = pd.DataFrame({
        "ID": lst["id_membro"], L("Safra", "Cohort"): lst["safra"], L("Mês de vida", "Month"): "M" + lst["mes_vida"].astype(str),
        L("Dias p/ renovar", "Days to renew"): lst["dias_para_renovar"],
        L("Prob. renovar", "Renewal prob."): lst["prob"], L("Faixa", "Band"): lst["faixa_risco"].map(tv),
        L("Pedidos", "Orders"): lst["pedidos_acum"], L("Atrasômetro", "Delay-meter"): lst["atrasometro"],
        L("Economia (R$)", "Savings (R$)"): lst["economia_acum"].round(2),
        L("Setor principal", "Main section"): lst["setor_principal"].map(tv),
        L("Ação sugerida", "Suggested action"): lst["acao_sugerida"].map(tv),
    })
    nv = f"{len(view):,}".replace(",", ".")
    st.caption(L(f"{nv} membros na seleção, do menor para o maior score.",
                 f"{nv} members selected, lowest score first."))
    st.dataframe(view.style.format({L("Prob. renovar", "Renewal prob."): "{:.1%}", L("Atrasômetro", "Delay-meter"): "{:+.0%}",
                                    L("Economia (R$)", "Savings (R$)"): "{:.2f}"}),
                 width="stretch", hide_index=True, height=420)
    st.download_button(L("Baixar lista (CSV)", "Download list (CSV)"), view.to_csv(index=False).encode("utf-8-sig"),
                       "penelope_lista_acao.csv", "text/csv")

# ================================================================== 7. Simulador
with abas[6]:
    st.markdown(L("Monte um membro e veja a chance de ele renovar, pelo GAM do mês de vida escolhido.",
                  "Build a member and see their renewal chance, using the GAM for the chosen month."))
    mods = modelos()
    c1, c2, c3 = st.columns(3)
    mes_s = c1.select_slider(L("Mês de vida", "Month of tenure"), MESES, value="M6")
    m_int = int(mes_s[1:])
    ped = c1.slider(NOMES["pedidos_acum"], 0, 20, min(m_int + 1, 20))
    ritmo = c1.slider(L("Ritmo de compra (a cada X dias)", "Buying rhythm (every X days)"), 7, 90, 30, disabled=ped < 2)
    dias_sem = c2.slider(L("Dias desde a última compra", "Days since last purchase"), 0, 30 * (m_int + 1),
                         min(20, 30 * (m_int + 1)), disabled=ped < 2)
    sem_ritmo = int(ped < 2)
    atraso = 0.0 if sem_ritmo else float(np.clip(dias_sem / ritmo - 1, -1, 6))
    if sem_ritmo:
        c2.markdown(L("**Atrasômetro:** sem ritmo ainda (precisa de 2+ pedidos)",
                      "**Delay-meter:** no rhythm yet (needs 2+ orders)"))
    else:
        c2.markdown(L(f"**Atrasômetro:** {dias_sem} ÷ {ritmo} − 1 = **{atraso:+.0%}**",
                      f"**Delay-meter:** {dias_sem} ÷ {ritmo} − 1 = **{atraso:+.0%}**"))
    ticket = c2.slider(NOMES["ticket_medio"], 0, 300, 110)
    econ_sug = int(ped * (0.05 * ticket + 10))
    econ = c2.slider(NOMES["economia_acum"], 0, 400, min(econ_sug, 400))
    setores = c3.slider(NOMES["qtd_setores"], 0, 6, min(3, max(ped, 0)))
    uso = c3.toggle(NOMES["uso_continuo"], value=False)
    canal = c3.selectbox(NOMES["canal"], ["App", "Site", "Loja"], format_func=tv)
    pag = c3.selectbox(NOMES["pagamento"], ["Cartão recorrente", "Pix/boleto"], format_func=tv)
    setor_p = c3.selectbox(NOMES["setor_principal"], ["Medicamentos", "Dermocosméticos", "Higiene e beleza",
                                                      "Infantil", "Suplementos", "Conveniência", "Sem compra"], format_func=tv)
    if ped == 0:
        ticket, econ, setores, setor_p = 0, 0, 0, "Sem compra"
    x = pd.DataFrame([dict(pedidos_acum=ped, atrasometro=atraso, ticket_medio=ticket, economia_acum=econ,
                           qtd_setores=setores, uso_continuo=int(uso), sem_ritmo=sem_ritmo, canal=canal, pagamento=pag,
                           setor_principal=setor_p)])[FEATURES]
    prob = float(mods[m_int].predict_proba(x)[0, 1])
    fx = "Alto risco" if prob < 0.4 else ("Médio risco" if prob < 0.7 else "Baixo risco")

    st.divider()
    r1, r2 = st.columns([1, 2])
    r1.markdown(f'<div class="kpi"><div class="v" style="font-size:2.6rem">{prob:.0%}</div>'
                f'<div class="l">{L("chance de renovar no M12", "chance of renewing at M12")}</div>'
                f'<div style="margin-top:8px;font-weight:600;color:{COR_RISCO[fx]}">● {tv(fx)}</div></div>',
                unsafe_allow_html=True)
    if ped == 0:
        r1.caption(L("Sem pedidos: ticket, economia e setores zerados.", "No orders: ticket, savings and sections set to zero."))

    if sem_ritmo:
        r2.info(L("Com menos de 2 pedidos o membro ainda não tem ritmo de compra, então o atrasômetro não entra. "
                  "Aumente os pedidos para ver o efeito do atraso.",
                  "With fewer than 2 orders the member has no buying rhythm yet, so the delay-meter is off. "
                  "Raise the orders to see the delay effect."))
    else:
        grid = np.arange(0, 30 * (m_int + 1) + 1, max(1, (30 * (m_int + 1)) // 40))
        xs = pd.concat([x] * len(grid), ignore_index=True)
        xs["atrasometro"] = np.clip(grid / ritmo - 1, -1, 6)
        ps = mods[m_int].predict_proba(xs[FEATURES])[:, 1]
        fig = go.Figure()
        fig.add_hrect(y0=0, y1=0.4, fillcolor="#FEE2E2", opacity=0.5, line_width=0, layer="below")
        fig.add_hrect(y0=0.4, y1=0.7, fillcolor="#FEF3C7", opacity=0.5, line_width=0, layer="below")
        fig.add_scatter(x=grid, y=ps, line=dict(color=AZUL, width=2.5), name="", showlegend=False,
                        hovertemplate=L("%{x} dias sem comprar: %{y:.0%}<extra></extra>", "%{x} days without buying: %{y:.0%}<extra></extra>"))
        fig.add_scatter(x=[dias_sem], y=[prob], mode="markers", marker=dict(size=13, color=AZUL_ESC, line=dict(color="white", width=2)),
                        showlegend=False, hoverinfo="skip")
        fig.update_yaxes(range=[0, 1], tickformat=".0%", title=L("chance de renovar", "renewal chance"))
        fig.update_xaxes(title=L("dias desde a última compra", "days since last purchase"))
        fig.update_layout(title=L("E se esse membro ficar mais tempo sem comprar?", "What if this member goes longer without buying?"))
        r2.plotly_chart(layout(fig, 320), width="stretch")

# ================================================================== 8. Metodologia
with abas[7]:
    st.markdown(L("""
### Como o modelo foi desenvolvido

**1. Base.** 49 mil membros sintéticos de um programa pago de drogaria, com cerca de 370 mil pedidos. A base foi gerada
com regras de comportamento realistas: cada membro tem um ritmo de compra próprio, e parte deles "se desliga" em algum
momento do ano (passa a comprar bem mais espaçado ou para de comprar). A renovação depende do que aconteceu no ano.

**2. Uma foto por mês de vida.** Para cada membro, tiramos 12 fotos: no fim do M0, do M1… até o M11. Cada foto usa só o
que tinha acontecido até aquele dia. O alvo é sempre o mesmo: **renovou no M12?** Assim o modelo do M3, por exemplo,
aprende a prever a renovação com três meses de informação.

**3. Variáveis.**
- **Pedidos acumulados**, **ticket médio**, **economia acumulada no programa** (frete grátis + cashback, ou seja, o uso do benefício);
- **Atrasômetro**: dias sem comprar ÷ ritmo médio do membro − 1. Ele cobre a recência, então não entra uma variável de
recência separada. **Cuidado com o começo:** o ritmo só existe com 2+ pedidos (é preciso ao menos um intervalo entre
compras). No M0, cerca de 3 em cada 4 membros têm 0 ou 1 compra. Em vez de inventar um ritmo de referência, esses
membros ficam marcados como **sem ritmo** (uma variável sim/não) e o atrasômetro entra neutro. É por isso que o
atrasômetro quase não pesa no M0 e vira a variável nº 1 a partir do meio do ano;
- **Quantidade de setores comprados** (medicamentos, dermocosméticos, higiene e beleza, infantil, suplementos, conveniência);
- **Comprou medicamento de uso contínuo** (sim/não);
- **Canal de adesão**, **forma de pagamento** e **setor principal**.

**4. Modelo: GAM logístico, um por mês.** Cada variável numérica vira uma curva suave (splines cúbicos) e as
categóricas viram efeitos fixos, tudo somado. A suavidade é controlada por penalização; o nível de penalização de cada
mês foi escolhido por validação cruzada de 5 partes no treino, pelo maior AUC.

**5. Validação.**
- **Treino × teste** (70/30, separando por membro, para o mesmo membro não aparecer nos dois lados);
- **Out-of-time**: aplicação nas safras de 2025, um período que o modelo nunca viu e com comportamento um pouco
diferente (renovação menor, mais adesões pelo app);
- **PSI**: compara a distribuição do score entre treino e out-of-time.

**6. Aplicação.** A carteira ativa é escorada pelo modelo do mês de vida em que cada membro está, e recebe faixa de risco
e uma ação sugerida.

### Glossário
| Termo | O que é |
|---|---|
| **Safra** | Grupo de membros que aderiu ao programa no mesmo mês. |
| **Mês de vida (M0…M11)** | Há quantos meses o membro está no programa. M0 = primeiro mês. |
| **Atrasômetro** | Quanto o membro está fora do próprio ritmo de compra. +50% = está 50% além do intervalo normal dele. Só existe com 2+ pedidos. |
| **Sem ritmo** | Membro com 0 ou 1 pedido: ainda não dá pra saber o ritmo dele. |
| **GAM** | Modelo aditivo generalizado: soma de curvas, uma por variável. Captura relações não lineares e continua fácil de explicar. |
| **Spline** | Curva flexível feita de pedaços de polinômio emendados suavemente. |
| **Logito** | Escala em que o GAM soma os efeitos. Positivo = mais chance de renovar; negativo = menos. |
| **AUC** | Probabilidade de o modelo dar nota maior a um membro que renovou do que a um que não renovou. 0,5 = sorteio; 1 = perfeito. |
| **Gini** | 2 × AUC − 1. Mesma informação da AUC, em outra escala. |
| **KS** | Maior distância entre as curvas acumuladas de quem renovou e de quem não renovou, ordenando pelo score. Quanto maior, mais o modelo separa os dois grupos. Régua usual: < 20 fraco, 20–30 aceitável, 30–40 bom, ≥ 40 muito bom. |
| **Decil** | A base ordenada pelo score e dividida em 10 partes iguais. |
| **Calibração** | Se o modelo diz 70%, cerca de 70% daqueles membros renovam de fato? |
| **Out-of-time** | Testar o modelo num período posterior ao do treino. |
| **PSI** | Índice de estabilidade populacional. Abaixo de 0,10 = estável; 0,10–0,25 = atenção; acima de 0,25 = mudou. |
| **Validação cruzada** | Dividir o treino em partes, treinar em umas e medir nas outras, para escolher parâmetros sem olhar o teste. |
""", """
### How the model was built

**1. Data.** 49k synthetic members of a paid drugstore program, with about 370k orders. The data was generated with
realistic behavior rules: each member has their own buying rhythm, and some of them "disengage" at some point in the
year (buying much less often or stopping). Renewal depends on what happened during the year.

**2. One snapshot per month of tenure.** Each member gets 12 snapshots: end of M0, M1… up to M11. Each snapshot only uses
what had happened up to that day. The target is always the same: **renewed at M12?** So the M3 model, for instance,
learns to predict renewal with three months of information.

**3. Variables.**
- **Cumulative orders**, **average ticket**, **cumulative program savings** (free shipping + cashback, i.e. benefit usage);
- **Delay-meter**: days without buying ÷ member's average rhythm − 1. It covers recency, so there is no separate recency
variable. **Careful at the start:** a rhythm only exists with 2+ orders (you need at least one gap between purchases).
At M0 about 3 in 4 members have 0 or 1 purchase. Instead of making up a reference rhythm, they are flagged as
**no rhythm** (a yes/no variable) and the delay-meter enters as neutral. That is why the delay-meter barely matters at
M0 and becomes the #1 variable from mid-year on;
- **Number of sections bought** (medicines, dermocosmetics, hygiene & beauty, baby & kids, supplements, convenience);
- **Bought continuous-use medication** (yes/no);
- **Sign-up channel**, **payment method** and **main section**.

**4. Model: logistic GAM, one per month.** Each numeric variable becomes a smooth curve (cubic splines) and the
categorical ones become fixed effects, all added up. Smoothness is controlled by a penalty; each month's penalty was
chosen by 5-fold cross-validation on the training set, by highest AUC.

**5. Validation.**
- **Train × test** (70/30, split by member so the same member never appears on both sides);
- **Out-of-time**: applied to the 2025 cohorts, a period the model never saw, with slightly different behavior
(lower renewal, more app sign-ups);
- **PSI**: compares the score distribution between train and out-of-time.

**6. Application.** The active base is scored by the model for each member's current month, and gets a risk band and
a suggested action.

### Glossary
| Term | Meaning |
|---|---|
| **Cohort** | Members who joined the program in the same month. |
| **Month of tenure (M0…M11)** | How many months the member has been in the program. M0 = first month. |
| **Delay-meter** | How far the member is off their own buying rhythm. +50% = 50% past their normal interval. Only exists with 2+ orders. |
| **No rhythm** | Member with 0 or 1 order: their rhythm can't be known yet. |
| **GAM** | Generalized additive model: a sum of curves, one per variable. Captures non-linear effects and stays easy to explain. |
| **Spline** | A flexible curve made of polynomial pieces joined smoothly. |
| **Logit** | The scale where the GAM adds the effects. Positive = more likely to renew; negative = less. |
| **AUC** | Probability that the model scores a renewing member higher than a non-renewing one. 0.5 = coin flip; 1 = perfect. |
| **Gini** | 2 × AUC − 1. Same information as AUC on another scale. |
| **KS** | Largest gap between the cumulative curves of renewers and non-renewers, sorted by score. Bigger = better separation. Usual scale: < 20 weak, 20–30 acceptable, 30–40 good, ≥ 40 very good. |
| **Decile** | The base sorted by score and split into 10 equal parts. |
| **Calibration** | When the model says 70%, do about 70% of those members actually renew? |
| **Out-of-time** | Testing the model on a period after the training period. |
| **PSI** | Population stability index. Below 0.10 = stable; 0.10–0.25 = watch; above 0.25 = shifted. |
| **Cross-validation** | Splitting training data into folds, training on some and scoring on others, to tune parameters without touching the test set. |
"""))
