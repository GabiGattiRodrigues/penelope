"""
Penélope — geração da base sintética do "Vida+" de uma rede de drogarias.

Cenário: programa de vantagens PAGO e ANUAL (R$ 119,90/ano) com frete grátis e cashback.
Cada membro compra ao longo de 12 meses e, no dia 365, renova ou não.

Três recortes de safra (mês de adesão):
  - desenvolvimento : jan/2024 a dez/2024  -> treino/teste
  - out-of-time     : jan/2025 a ago/2025  -> validação fora do tempo (renovação já conhecida)
  - carteira ativa  : set/2025 a ago/2026  -> ainda não chegaram no M12: é onde o modelo é aplicado

Data de corte: 31/08/2026.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
DATA_CORTE = pd.Timestamp("2026-08-31")
MENSALIDADE = 119.90
FRETE = 14.90
CASHBACK = 0.05
SETORES = ["Medicamentos", "Dermocosméticos", "Higiene e beleza",
           "Infantil", "Suplementos", "Conveniência"]
ADESOES_POR_MES = 1500

OUT = Path(__file__).resolve().parents[1] / "data"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gerar():
    rng = np.random.default_rng(SEED)
    safras = pd.period_range("2024-01", "2026-08", freq="M")

    # ---------------- membros ----------------
    linhas = []
    mid = 0
    for s in safras:
        n = int(ADESOES_POR_MES * rng.uniform(0.85, 1.15))
        ini = s.to_timestamp()
        dias_mes = s.days_in_month
        ano25 = s.year >= 2025
        for _ in range(n):
            mid += 1
            p_canal = [0.55, 0.22, 0.23] if ano25 else [0.45, 0.25, 0.30]
            linhas.append(dict(
                id_membro=mid,
                safra=str(s),
                data_adesao=ini + pd.Timedelta(days=int(rng.integers(0, dias_mes))),
                canal=rng.choice(["App", "Site", "Loja"], p=p_canal),
                pagamento=rng.choice(["Cartão recorrente", "Pix/boleto"], p=[0.6, 0.4]),
            ))
    m = pd.DataFrame(linhas)
    n = len(m)
    ano25 = m["safra"].str[:4].astype(int).ge(2025).to_numpy()

    z = rng.normal(0, 1, n)                           # engajamento latente
    paciente = rng.random(n) < 0.33                   # usa medicamento de uso contínuo
    recorrente = (m["pagamento"] == "Cartão recorrente").to_numpy()
    online = (m["canal"] != "Loja").to_numpy()

    # ritmo de compra de cada membro
    intervalo = np.where(paciente,
                         rng.normal(30, 4, n).clip(20, 45),
                         np.exp(rng.normal(np.log(38), 0.35, n)).clip(14, 90))
    forma_gamma = np.where(paciente, 7.0, 2.0)        # paciente é mais regular

    # desengajamento (o cliente "sai do comportamento dele")
    p_des = sigmoid(-0.2 - 0.9 * z - 0.9 * paciente - 0.4 * recorrente + 0.25 * ano25)
    desengaja = rng.random(n) < p_des
    # quem se desliga tende a se desligar cedo (1º semestre), mas pode acontecer no ano todo
    dia_des = np.where(desengaja, 15 + 345 * rng.beta(1.2, 2.4, n), 999)
    para_tudo = rng.random(n) < 0.55                  # depois de desengajar: para de vez ou espaça

    # amplitude de setores: preferência de setor por membro
    amplitude = sigmoid(0.9 * z + 0.3 * rng.normal(size=n))  # 0..1
    base_pref = np.array([1.0, 0.7, 1.0, 0.35, 0.45, 0.6])

    # ---------------- pedidos ----------------
    ped = []
    for i in range(n):
        pref = base_pref * np.exp(rng.normal(0, 0.8, 6))
        if paciente[i]:
            pref[0] *= 4
        pref = pref / pref.sum()
        t = rng.exponential(intervalo[i] * 0.5)       # 1ª compra costuma vir cedo
        while t < 365:
            if t > dia_des[i]:
                if para_tudo[i]:
                    break
                passo_mult = 3.2
            else:
                passo_mult = 1.0
            k = 1 + rng.binomial(2, 0.15 + 0.5 * amplitude[i])
            setores = rng.choice(6, size=k, replace=False, p=pref)
            mask = int(sum(1 << int(s) for s in setores))
            uso_cont = bool((paciente[i] and rng.random() < 0.85) or rng.random() < 0.03)
            if uso_cont:
                mask |= 1
            ticket = float(np.exp(rng.normal(np.log(125 if paciente[i] else 85), 0.45)) * (1 + 0.25 * (k - 1)))
            canal_ped_online = online[i] if rng.random() < 0.85 else not online[i]
            ped.append((m.at[i, "id_membro"], int(t), round(ticket, 2), mask, uso_cont, canal_ped_online))
            t += rng.gamma(forma_gamma[i], intervalo[i] * passo_mult / forma_gamma[i])
    p = pd.DataFrame(ped, columns=["id_membro", "dia", "ticket", "setores_mask", "uso_continuo", "online"])
    p["economia"] = (p["online"] * FRETE + CASHBACK * p["ticket"]).round(2)

    # ---------------- renovação no dia 365 ----------------
    agg = p.groupby("id_membro").agg(pedidos=("dia", "size"), ultimo=("dia", "max"),
                                     economia=("economia", "sum"),
                                     mask=("setores_mask", lambda s: np.bitwise_or.reduce(s.to_numpy())))
    agg = agg.reindex(m["id_membro"]).fillna({"pedidos": 0, "ultimo": -999, "economia": 0, "mask": 0})
    n_set = agg["mask"].astype(int).apply(lambda x: bin(x).count("1")).to_numpy()
    ativo_fim = (365 - agg["ultimo"].to_numpy()) < np.maximum(1.6 * intervalo, 45)
    frac_engajado = np.minimum(dia_des, 365) / 365
    logit = (-3.6 + 1.2 * frac_engajado + 1.9 * ativo_fim + 0.45 * np.log1p(agg["pedidos"].to_numpy())
             + 0.55 * recorrente + 0.45 * np.minimum(agg["economia"].to_numpy() / MENSALIDADE, 2.5)
             + 0.18 * (n_set - 1) + 0.35 * paciente + 0.35 * z
             - 0.15 * ano25 + rng.normal(0, 0.6, n))
    m["renovou"] = (rng.random(n) < sigmoid(logit)).astype(int)

    # datas de pedido e corte temporal
    p = p.merge(m[["id_membro", "data_adesao"]], on="id_membro")
    p["data_pedido"] = p["data_adesao"] + pd.to_timedelta(p["dia"], unit="D")
    p = p[p["data_pedido"] <= DATA_CORTE].drop(columns="data_adesao")

    m["data_decisao"] = m["data_adesao"] + pd.Timedelta(days=365)
    m["recorte"] = np.select(
        [m["safra"] <= "2024-12", m["safra"] <= "2025-08"],
        ["desenvolvimento", "out-of-time"], "carteira ativa")
    m.loc[m["recorte"] == "carteira ativa", "renovou"] = np.nan  # ainda não sabemos

    OUT.mkdir(parents=True, exist_ok=True)
    m.to_parquet(OUT / "membros.parquet", index=False)
    p.to_parquet(OUT / "pedidos.parquet", index=False)
    print(m.groupby("recorte").agg(membros=("id_membro", "size"), renovacao=("renovou", "mean")))
    print("pedidos:", len(p), "| pedidos/membro:", round(len(p) / n, 1))


if __name__ == "__main__":
    gerar()
