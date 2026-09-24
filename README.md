# Penélope · quem vai continuar fiel?

Previsão de **renovação de um programa de fidelidade pago e anual** (cenário: rede de drogarias, "Vida+"),
a partir de cada mês de vida do membro, do **M0 ao M11**, com **12 modelos GAM**.

App: https://penelopechurn.streamlit.app/

Reconstrução pública, com **dados 100% sintéticos**, de um projeto real que fiz no varejo.

## O que tem aqui
- **Base sintética**: ~49 mil membros, ~390 mil pedidos, 32 safras (jan/2024 a ago/2026).
- **Variáveis por mês de vida**: pedidos acumulados, **atrasômetro** (dias sem comprar ÷ ritmo do cliente − 1,
  que cobre a recência; só existe com 2+ pedidos, antes disso o membro fica marcado como "sem ritmo"), ticket médio, economia no programa (uso do benefício), quantidade de setores,
  compra de medicamento de uso contínuo, canal, forma de pagamento e setor principal.
- **Modelo**: GAM logístico (B-splines cúbicos + efeitos fixos, penalizado), um por mês, com a penalização escolhida por CV.
- **Validação**: treino/teste (safras 2024), **out-of-time** (safras jan–ago/2025) e PSI.
  AUC out-of-time vai de 0,62 no M0 a 0,87 no M11; KS de 17 a 61 (acima de 40 a partir do M5).
- **Aplicação**: carteira ativa (safras set/2025–ago/2026) escorada, com faixa de risco e ação sugerida.
- **App Streamlit (PT/EN)**: caso, análise exploratória (distribuições por foto M0…M11, correlação), **mapa de calor variável × mês de vida × renovação**, performance
  (AUC, KS, Gini, ROC, curva KS, decis, calibração, PSI), efeitos do GAM, lista de ação e simulador.

## Rodar no Windows
Duplo clique em `rodar_penelope.bat` (na 1ª vez cria o ambiente e instala tudo).
Para regerar a base e retreinar: `retreinar.bat`.

## Rodar em qualquer SO
```bash
pip install -r requirements.txt
python src/gerar_base.py && python src/variaveis.py && python src/treinar.py   # opcional: já vem treinado
streamlit run app.py
```

## Estrutura
```
app.py                  app Streamlit
src/gerar_base.py       base sintética (membros e pedidos)
src/variaveis.py        fotos M0..M11 e carteira ativa
src/gam.py              definição do GAM
src/treinar.py          treino, métricas, bivariada e escoragem
data/                   base gerada
artefatos/              modelos, métricas e scores
assets/penelope.svg     identidade visual
```
