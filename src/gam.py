"""Definição do GAM logístico usado na Penélope (importado pelo treino e pelo app)."""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression

from variaveis import NUM, BIN, CAT

# limites para as curvas (evita que outliers dobrem as splines)
CLIP = {"pedidos_acum": (0, 20), "atrasometro": (-1, 4), "ticket_medio": (0, 300),
        "economia_acum": (0, 400), "qtd_setores": (0, 6)}


def _clip(X):
    X = X.copy()
    for c, (a, b) in CLIP.items():
        X[c] = X[c].clip(a, b)
    return X


def novo_gam(C):
    spl = SplineTransformer(n_knots=5, degree=3, knots="uniform", extrapolation="constant")
    pre = ColumnTransformer([
        ("s", spl, NUM),
        ("b", "passthrough", BIN),
        ("c", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("clip", FunctionTransformer(_clip)), ("pre", pre),
                     ("lr", LogisticRegression(C=C, max_iter=3000))])


