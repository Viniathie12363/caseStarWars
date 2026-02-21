# 🧬 Projeto Kamino: Classificação de Clones com Machine Learning

Este repositório contém um modelo de **Árvore de Decisão** desenvolvido para classificar o "Status" de clones em um dataset temático de Star Wars. O projeto foca em **interpretabilidade de modelos** e **preparação de dados (Data Wrangling)**.

## 📋 Sobre o Case
O desafio consiste em analisar métricas biométricas (massa, estatura, medidas de crânio, etc.) para determinar a classificação funcional de cada unidade. O foco técnico foi transformar um processo de decisão complexo em uma visualização lógica e espaçada.

## 🛠️ Stack Tecnológica
* **Linguagem:** Python
* **Bibliotecas:** * `pandas` para manipulação de dados.
    * `scikit-learn` para modelagem preditiva.
    * `matplotlib` para visualização de dados.
    * `pyarrow` para suporte a arquivos Parquet.

## 🚀 O que foi implementado
1.  **Tratamento de Dados:** * Limpeza de espaços vazios em nomes de colunas via `.str.strip()`.
    * Arredondamento de precisão na coluna `Massa(em kilos)` para 2 casas decimais.
    * Conversão de variáveis categóricas ("Tipo 1", "Tipo 2") para valores numéricos.
    * Tratamento de valores ausentes (NaN) com `fillna(0)`.
2.  **Modelagem:** * Uso do `DecisionTreeClassifier` com controle de profundidade (`max_depth=3`) para evitar overfitting e garantir a legibilidade.
3.  **Visualização:** * Customização do `plot_tree` com `figsize=(40, 15)` para garantir que os nós e setas fiquem bem espaçados e legíveis.

## 📊 Visualização do Modelo
A árvore gerada permite entender os critérios de decisão de forma visual:

