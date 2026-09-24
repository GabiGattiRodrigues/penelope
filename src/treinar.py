"""
Penélope — treino dos 12 GAMs (um por mês de vida), validação e aplicação.

GAM logístico: cada variável numérica entra como uma curva suave (B-splines cúbicos)
e as categóricas como efeitos fixos; tudo somado na escala do logito e penalizado (L2),
o que controla a "ondulação" das curvas. Por ser aditivo, dá pra desenhar o efeito de
cada variável separadamente — a mesma leitura do GAM clássico.

Fluxo:
  safras 2024 (desenvolvimento) -> 70% treino / 30% teste (split por membro)
  safras jan-ago/2025           -> out-of-time (aplicação numa base que o modelo nunca viu)
  carteira ativa                -> escoragem e lista de ação
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve

from variaveis import NUM, BIN, CAT, FEATURES
from gam import novo_gam, CLIP

RAIZ = Path(__file__).resolve().parents[1]
DATA, ART = RAIZ / "data", RAIZ / "artefatos"
ART.mkdir(exist_ok=True)
SEED = 7
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]

def ks(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def psi(ref, novo, cortes):
    a = np.histogram(ref, cortes)[0] / len(ref)
    b = np.histogram(novo, cortes)[0] / len(novo)
    a, b = np.clip(a, 1e-4, None), np.clip(b, 1e-4, None)
    return float(np.sum((b - a) * np.log(b / a)))


def cortes_decis(p_treino):
    c = np.quantile(p_treino, np.linspace(0, 1, 11))
    c[0], c[-1] = -np.inf, np.inf
    return np.unique(c)


def tabela_decis(y, p, cortes):
    d = pd.DataFrame({"y": y, "p": p})
    d["decil"] = pd.cut(d["p"], cortes, labels=False, include_lowest=True)
    d["decil"] = d["decil"].max() - d["decil"] + 1  # 1 = maior prob. de renovar
    t = d.groupby("decil").agg(n=("y", "size"), renovou=("y", "sum"), prob_media=("p", "mean")).reset_index()
    t["nao_renovou"] = t["n"] - t["renovou"]
    t["taxa_renovacao"] = t["renovou"] / t["n"]
    t["pct_renov_acum"] = t["renovou"].cumsum() / t["renovou"].sum()
    t["pct_nao_renov_acum"] = t["nao_renovou"].cumsum() / t["nao_renovou"].sum()
    t["ks"] = (t["pct_renov_acum"] - t["pct_nao_renov_acum"]).abs()
    return t.round(4).to_dict(orient="list")


def curva_ks(y, p, pontos=50):
    ordem = np.argsort(-p)
    y = np.asarray(y)[ordem]
    acum_r = np.cumsum(y) / y.sum()
    acum_n = np.cumsum(1 - y) / (1 - y).sum()
    idx = np.linspace(0, len(y) - 1, pontos).astype(int)
    return {"pop": ((idx + 1) / len(y)).round(4).tolist(), "renovou": acum_r[idx].round(4).tolist(),
            "nao_renovou": acum_n[idx].round(4).tolist()}


def curva_roc(y, p, pontos=60):
    fpr, tpr, _ = roc_curve(y, p)
    idx = np.unique(np.linspace(0, len(fpr) - 1, pontos).astype(int))
    return {"fpr": fpr[idx].round(4).tolist(), "tpr": tpr[idx].round(4).tolist()}


def calibracao(y, p):
    d = pd.DataFrame({"y": y, "p": p})
    d["b"] = pd.qcut(d["p"], 10, labels=False, duplicates="drop")
    t = d.groupby("b").agg(prevista=("p", "mean"), observada=("y", "mean"))
    return t.round(4).to_dict(orient="list")


def formas(modelo, Xtr, rng):
    """Efeito parcial de cada variável na escala do logito (centrado na média do treino)."""
    ref = {c: Xtr[c].median() for c in NUM + BIN}
    ref.update({c: Xtr[c].mode()[0] for c in CAT})
    amostra = Xtr.sample(min(3000, len(Xtr)), random_state=1)
    base = pd.DataFrame([ref])
    f0 = modelo.decision_function(base[FEATURES])[0]
    out, imp = {}, {}
    for v in FEATURES:
        # contribuição nos valores observados -> importância e centralização
        obs = pd.concat([base] * len(amostra), ignore_index=True)
        obs[v] = amostra[v].to_numpy()
        contrib = modelo.decision_function(obs[FEATURES]) - f0
        centro = contrib.mean()
        imp[v] = float(np.std(contrib))
        if v in NUM:
            a, b = CLIP[v]
            lo, hi = max(a, Xtr[v].quantile(0.01)), min(b, Xtr[v].quantile(0.99))
            if v == "qtd_setores":
                grid = np.arange(0, 7)
            else:
                grid = np.linspace(lo, hi, 60)
            g = pd.concat([base] * len(grid), ignore_index=True)
            g[v] = grid
            y = modelo.decision_function(g[FEATURES]) - f0 - centro
            out[v] = {"x": np.round(grid, 3).tolist(), "y": np.round(y, 4).tolist()}
        else:
            niveis = sorted(Xtr[v].unique().tolist())
            g = pd.concat([base] * len(niveis), ignore_index=True)
            g[v] = niveis
            y = modelo.decision_function(g[FEATURES]) - f0 - centro
            out[v] = {"x": [str(n) for n in niveis], "y": np.round(y, 4).tolist()}
    return out, imp


# ---------------- faixas para a análise bivariada ----------------
FAIXAS = {
    "pedidos_acum": ([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 8.5, 10.5, 12.5, 15.5, 19.5, np.inf],
                     ["0", "1", "2", "3", "4", "5–6", "7–8", "9–10", "11–12", "13–15", "16–19", "20+"]),
    "atrasometro": ([-np.inf, -0.5, 0, 0.25, 0.5, 1, 2, np.inf],
                    ["≤ −50%", "−50% a 0%", "0% a 25%", "25% a 50%", "50% a 100%", "100% a 200%", "> 200%"]),
    "ticket_medio": ([-np.inf, 0.01, 60, 90, 120, 160, 220, np.inf],
                     ["Sem compra", "< R$60", "R$60–90", "R$90–120", "R$120–160", "R$160–220", "R$220+"]),
    "economia_acum": ([-np.inf, 0.01, 25, 50, 100, 150, 200, 300, np.inf],
                      ["R$0", "< R$25", "R$25–50", "R$50–100", "R$100–150", "R$150–200", "R$200–300", "R$300+"]),
    "qtd_setores": ([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], ["0", "1", "2", "3", "4", "5", "6"]),
}
SEM_RITMO = "Sem ritmo (< 2 pedidos)"


def bivariada(df):
    linhas = []
    for v in FEATURES:
        if v == "atrasometro":
            cortes, rot = FAIXAS[v]
            fx = pd.cut(df[v], cortes, labels=rot).astype(str).where(df["sem_ritmo"] == 0, SEM_RITMO)
            ordem = {r: i + 1 for i, r in enumerate(rot)}
            ordem[SEM_RITMO] = 0
        elif v in FAIXAS:
            cortes, rot = FAIXAS[v]
            fx = pd.cut(df[v], cortes, labels=rot)
            ordem = {r: i for i, r in enumerate(rot)}
        elif v in ("uso_continuo", "sem_ritmo"):
            fx = df[v].map({0: "Não", 1: "Sim"})
            ordem = {"Não": 0, "Sim": 1}
        else:
            fx = df[v]
            ordem = {r: i for i, r in enumerate(sorted(df[v].unique()))}
        t = (df.assign(faixa=fx.astype(str)).groupby(["mes_vida", "faixa"], observed=True)
             .agg(n=("renovou", "size"), taxa=("renovou", "mean"), prob=("prob", "mean")).reset_index())
        t["variavel"] = v
        t["ordem"] = t["faixa"].map(ordem)
        linhas.append(t)
    return pd.concat(linhas, ignore_index=True)


def main():
    rng = np.random.default_rng(SEED)
    fotos = pd.read_parquet(DATA / "fotos_historico.parquet")
    cart = pd.read_parquet(DATA / "carteira_ativa.parquet")

    dev_ids = fotos.loc[fotos["recorte"] == "desenvolvimento", "id_membro"].unique()
    teste_ids = set(rng.choice(dev_ids, int(0.3 * len(dev_ids)), replace=False))
    fotos["amostra"] = np.where(fotos["recorte"] == "out-of-time", "out-of-time",
                                np.where(fotos["id_membro"].isin(teste_ids), "teste", "treino"))

    modelos, metr, scores = {}, {"meses": {}}, []
    cart["prob"] = np.nan
    for m in range(12):
        f = fotos[fotos["mes_vida"] == m]
        tr = f[f["amostra"] == "treino"]
        Xtr, ytr = tr[FEATURES], tr["renovou"].astype(int)

        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        cv_auc = {C: cross_val_score(novo_gam(C), Xtr, ytr, cv=cv, scoring="roc_auc").mean() for C in C_GRID}
        C = max(cv_auc, key=cv_auc.get)
        gam = novo_gam(C).fit(Xtr, ytr)
        modelos[m] = gam

        f = f.assign(prob=gam.predict_proba(f[FEATURES])[:, 1])
        scores.append(f)
        cortes = cortes_decis(f.loc[f["amostra"] == "treino", "prob"])

        mm = {"C": C, "cv_auc": {str(k): round(v, 4) for k, v in cv_auc.items()}, "amostras": {}}
        for a in ["treino", "teste", "out-of-time"]:
            s = f[f["amostra"] == a]
            y, p = s["renovou"].astype(int).to_numpy(), s["prob"].to_numpy()
            auc = roc_auc_score(y, p)
            mm["amostras"][a] = {
                "n": int(len(s)), "taxa_renovacao": round(float(y.mean()), 4),
                "auc": round(auc, 4), "gini": round(2 * auc - 1, 4), "ks": round(ks(y, p), 4),
                "decis": tabela_decis(y, p, cortes), "curva_ks": curva_ks(y, p),
                "roc": curva_roc(y, p), "calibracao": calibracao(y, p),
            }
        ptr = f.loc[f["amostra"] == "treino", "prob"]
        mm["psi_oot"] = round(psi(ptr, f.loc[f["amostra"] == "out-of-time", "prob"], cortes), 4)

        idx = cart["mes_vida"] == m
        if idx.any():
            cart.loc[idx, "prob"] = gam.predict_proba(cart.loc[idx, FEATURES])[:, 1]
            mm["psi_carteira"] = round(psi(ptr, cart.loc[idx, "prob"], cortes), 4)
        mm["formas"], mm["importancia"] = formas(gam, Xtr, rng)
        metr["meses"][str(m)] = mm
        a = mm["amostras"]
        print(f"M{m:<2} C={C:<5} AUC tr/te/oot = {a['treino']['auc']:.3f}/{a['teste']['auc']:.3f}/"
              f"{a['out-of-time']['auc']:.3f}  KS = {a['treino']['ks']:.3f}/{a['teste']['ks']:.3f}/"
              f"{a['out-of-time']['ks']:.3f}  PSI oot={mm['psi_oot']:.3f}")

    scores = pd.concat(scores, ignore_index=True)
    biv = bivariada(scores[scores["amostra"] != "out-of-time"])

    # ---------------- lista de ação da carteira ativa ----------------
    cart["faixa_risco"] = pd.cut(cart["prob"], [-0.01, 0.4, 0.7, 1.01],
                                 labels=["Alto risco", "Médio risco", "Baixo risco"]).astype(str)
    cart["dias_para_renovar"] = (cart["data_decisao"] - pd.Timestamp("2026-08-31")).dt.days

    def acao(r):
        if r["faixa_risco"] == "Alto risco":
            if r["pedidos_acum"] == 0:
                return "Ativação: 1ª compra com cupom do programa"
            if r["atrasometro"] > 0.5:
                return "Resgate: lembrete de recompra + oferta no setor principal"
            return "Mostrar economia: extrato do que o programa já devolveu"
        if r["faixa_risco"] == "Médio risco":
            if r["atrasometro"] > 0.5:
                return "Lembrete de recompra no ritmo do cliente"
            return "Cross-sell: apresentar um setor novo"
        return "Manter: régua padrão + convite para renovação antecipada"

    cart["acao_sugerida"] = cart.apply(acao, axis=1)

    joblib.dump(modelos, ART / "modelos_gam.joblib", compress=3)
    (ART / "metricas.json").write_text(json.dumps(metr, ensure_ascii=False))
    biv.to_parquet(ART / "bivariada.parquet", index=False)
    cols = ["id_membro", "safra", "mes_vida", "amostra", "renovou", "prob"] + FEATURES
    scores[cols].to_parquet(ART / "scores_historico.parquet", index=False)
    cart.to_parquet(ART / "carteira_escorada.parquet", index=False)
    print(cart["faixa_risco"].value_counts().to_dict())


if __name__ == "__main__":
    main()
