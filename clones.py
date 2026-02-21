#%%
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

df = pd.read_parquet("dados_clones.parquet")
df.columns = df.columns.str.strip()

#%%
features = ['Massa(em kilos)', 'Estatura(cm)', 'Distância Ombro a ombro', 'Tamanho do crânio', 'Tamanho dos pés']
target = 'Status'

X = df[features].copy()
y = df[target]

X['Massa(em kilos)'] = X['Massa(em kilos)'].round(2)
mapa_tipos = {"Tipo 1": 1, "Tipo 2": 2, "Tipo 3": 3, "Tipo 4": 4, "Tipo 5": 5}
X = X.replace(mapa_tipos).fillna(0)

#%%
model = tree.DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

#%%
plt.figure(figsize=(40, 15))

tree.plot_tree(
    model,
    feature_names=features,
    class_names=[str(c) for c in model.classes_],
    filled=True,
    rounded=True,
    precision=2,
    fontsize=14
)

plt.show()

