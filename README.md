# MVP - Otimização de Preços no Varejo com Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/renatavirginia/machine-learning/blob/master/retail_price_optimization.ipynb)

## Sobre o Projeto

Este projeto constrói um pipeline completo de Machine Learning para **otimização de preços no varejo**. Usando dados reais de e-commerce brasileiro, treinamos modelos de regressão supervisionada para prever a demanda em função do preço e, com base nessas previsões, simulamos diferentes cenários para encontrar o ponto de preço que maximiza a receita, calculada como **Receita = Preço × Demanda Prevista**.

## Hipóteses Investigadas

1. Existe uma relação negativa entre o preço unitário e a quantidade vendida? Produtos mais caros tendem a vender menos?
2. Modelos de ensemble como Random Forest e XGBoost superam a Regressão Linear porque a relação preço-demanda não é perfeitamente linear?
3. Os preços dos concorrentes (`comp_1`, `comp_2`, `comp_3`) ajudam a prever a demanda ao capturar o posicionamento relativo de preço no mercado?
4. O preço praticado está abaixo do preço ótimo para a maioria dos produtos, indicando uma oportunidade real de reajuste?

## Dataset

| Informação | Detalhe |
|---|---|
| **Fonte** | [Retail Price Optimization (Kaggle)](https://www.kaggle.com/datasets/suddharshan/retail-price-optimization/data) |
| **Origem** | Dados de vendas de e-commerce brasileiro |
| **Atributos** | 24 variáveis (preços, concorrência, produto, sazonalidade) |
| **Tipo** | Dados reais de transações comerciais |
| **Variável Alvo** | `qty`, a quantidade de unidades vendidas |

As variáveis cobrem preços próprios e de concorrentes, características do produto (peso, volume, avaliação, fotos), indicadores temporais (mês, ano, estação, feriados) e métricas de clientes.

## Estrutura da Análise

1. **Descrição do Problema**: contexto de negócio, hipóteses, tipo de problema, seleção de dados e dicionário de atributos
2. **Importação e Carga de Dados**: configuração do ambiente e carregamento direto via URL pública do GitHub
3. **Análise Exploratória (EDA)**: estatísticas descritivas, valores faltantes, distribuições, boxplots, comportamento do preço e matriz de correlação
4. **Feature Engineering**: criação de variáveis de posicionamento competitivo (`price_vs_competition`, `price_ratio`, `comp_price_mean`)
5. **Pré-Processamento**: pipeline com imputação por mediana/moda, winsorização (IQR × 3), OneHotEncoding, StandardScaler e divisão 80/20
6. **Modelagem**: três modelos em ordem crescente de complexidade: Regressão Linear (baseline), Random Forest (com tuning via RandomizedSearchCV) e XGBoost
7. **Avaliação dos Modelos**: métricas MAE, RMSE, R², MAPE, análise de overfitting e gráficos Real vs Previsto
8. **Simulação e Otimização de Receita**: 150 pontos de preço simulados por produto para identificar o preço que maximiza a receita
9. **Conclusão**: painéis visuais consolidados, respostas às hipóteses, limitações e próximos passos

## Principais Resultados

- A relação **Preço vs Demanda** confirmou a lei da demanda: correlação negativa entre `unit_price` e `qty`
- O **Random Forest Otimizado** (RandomizedSearchCV) apresentou o melhor desempenho geral, superando a regressão linear em todas as métricas
- A variável `unit_price` foi o **principal driver de demanda** na análise de importância de features
- O **preço ótimo ficou entre 50% e 80%** do preço máximo observado por produto, com potencial de aumento de receita em todos os segmentos
- O modelo captura a **elasticidade-preço** da demanda, permitindo precificação baseada em dados no lugar do markup fixo

## Tecnologias

| Categoria | Ferramentas |
|---|---|
| Manipulação de Dados | Python, Pandas, NumPy |
| Visualização | Matplotlib, Seaborn |
| Pré-Processamento | Scikit-learn (Pipeline, ColumnTransformer, StandardScaler, OneHotEncoder) |
| Modelagem | Scikit-learn (LinearRegression, RandomForestRegressor), XGBoost |
| Otimização | RandomizedSearchCV (5-fold cross-validation) |
