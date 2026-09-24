"""
Penélope — variáveis por mês de vida (M0 a M11).

A "foto" do mês m é tirada no fim do mês de vida m (dia 30*(m+1) desde a adesão),
usando só os pedidos feitos até ali. O alvo é sempre o mesmo: renovou no M12?

Não existe variável de recência isolada: o ATRASÔMETRO cobre essa janela,
comparando o tempo sem comprar com o ritmo do próprio cliente.

Cuidado com o começo da vida: o ritmo só existe a partir de 2 pedidos (é preciso ao menos
um intervalo entre compras). Com 0 ou 1 pedido o membro fica marcado como SEM RITMO
(sem_ritmo = 1) e o atrasômetro entra neutro (0), em vez de inventar um ritmo de referência.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
SETORES = ["Medicamentos", "Dermocosméticos", "Higiene e beleza",
           "Infantil", "Suplementos", "Conveniência"]
DATA_CORTE = pd.Timestamp("2026-08-31")

NUM = ["pedidos_acum", "atrasometro", "ticket_medio", "economia_acum", "qtd_setores"]
BIN = ["uso_continuo", "sem_ritmo"]
CAT = ["canal", "pagamento", "setor_principal"]
FEATURES = NUM + BIN + CAT


def _popcount(x):
    x = np.asarray(x, dtype=np.int64)
    return sum(((x >> k) & 1) for k in range(6))


def foto(membros: pd.DataFrame, pedidos: pd.DataFrame, dia_corte: pd.Series, mes: pd.Series) -> pd.DataFrame:
    """Calcula as variáveis de cada membro usando os pedidos com dia < dia_corte (por membro)."""
    base = membros[["id_membro", "safra", "recorte", "canal", "pagamento", "renovou"]].copy()
    base["dia_corte"] = dia_corte.to_numpy()
    base["mes_vida"] = mes.to_numpy()

    p = pedidos.merge(base[["id_membro", "dia_corte"]], on="id_membro")
    p = p[p["dia"] < p["dia_corte"]]

    # setor principal = setor que mais aparece nos pedidos
    cont = np.stack([((p["setores_mask"].to_numpy() >> k) & 1) for k in range(6)], axis=1)
    cont = pd.DataFrame(cont, columns=SETORES)
    cont["id_membro"] = p["id_membro"].to_numpy()
    cont = cont.groupby("id_membro").sum()
    principal = cont.idxmax(axis=1).rename("setor_principal")

    g = p.groupby("id_membro").agg(
        pedidos_acum=("dia", "size"), primeiro=("dia", "min"), ultimo=("dia", "max"),
        ticket_medio=("ticket", "mean"), economia_acum=("economia", "sum"),
        uso_continuo=("uso_continuo", "max"),
        mask=("setores_mask", lambda s: np.bitwise_or.reduce(s.to_numpy())),
    )
    g["qtd_setores"] = _popcount(g["mask"].to_numpy())
    g = g.join(principal)

    f = base.merge(g.drop(columns="mask"), left_on="id_membro", right_index=True, how="left")
    f["pedidos_acum"] = f["pedidos_acum"].fillna(0).astype(int)
    f["ticket_medio"] = f["ticket_medio"].fillna(0.0)
    f["economia_acum"] = f["economia_acum"].fillna(0.0)
    f["qtd_setores"] = f["qtd_setores"].fillna(0).astype(int)
    f["uso_continuo"] = f["uso_continuo"].fillna(False).astype(int)
    f["setor_principal"] = f["setor_principal"].fillna("Sem compra")

    # ---- atrasômetro (só com ritmo definido: >= 2 pedidos) ----
    tem_ritmo = f["pedidos_acum"] >= 2
    ritmo = ((f["ultimo"] - f["primeiro"]) / (f["pedidos_acum"] - 1).clip(lower=1)).clip(lower=7)
    dias_sem = np.where(f["pedidos_acum"] >= 1, f["dia_corte"] - f["ultimo"], f["dia_corte"])
    f["sem_ritmo"] = (~tem_ritmo).astype(int)
    f["ritmo_dias"] = np.where(tem_ritmo, ritmo.round(1), np.nan)
    f["dias_sem_compra"] = dias_sem.astype(int)
    f["atrasometro"] = np.where(tem_ritmo, np.clip(dias_sem / ritmo - 1, -1, 6), 0.0).round(3)

    return f.drop(columns=["primeiro", "ultimo"])


def montar():
    m = pd.read_parquet(DATA / "membros.parquet")
    p = pd.read_parquet(DATA / "pedidos.parquet")

    # ---- fotos M0..M11 para desenvolvimento e out-of-time ----
    hist = m[m["recorte"] != "carteira ativa"].reset_index(drop=True)
    fotos = []
    for mes in range(12):
        dc = pd.Series(np.full(len(hist), 30 * (mes + 1)))
        fotos.append(foto(hist, p, dc, pd.Series(np.full(len(hist), mes))))
    fotos = pd.concat(fotos, ignore_index=True)

    # ---- carteira ativa: uma foto na data de corte, no mês de vida atual ----
    at = m[m["recorte"] == "carteira ativa"].reset_index(drop=True)
    dias = (DATA_CORTE - at["data_adesao"]).dt.days + 1
    mes = np.clip(np.ceil(dias / 30) - 1, 0, 11).astype(int)
    cart = foto(at, p, dias, pd.Series(mes))
    cart = cart.merge(at[["id_membro", "data_adesao", "data_decisao"]], on="id_membro")

    fotos.to_parquet(DATA / "fotos_historico.parquet", index=False)
    cart.to_parquet(DATA / "carteira_ativa.parquet", index=False)
    print(fotos.groupby("mes_vida")[["pedidos_acum", "atrasometro", "qtd_setores"]].mean().round(2))
    print("carteira:", len(cart), cart["mes_vida"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    montar()
