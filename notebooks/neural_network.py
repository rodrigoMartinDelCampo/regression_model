import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "insurance.csv")

df = pd.read_csv(DATA_PATH)

# cargar y preparar el dataset
# leemos csv, codifico variables categóricas, separo x/y,
# convertimos tipos, normalizamos las features y las pasamos a tensores para pytorch.
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# codificamos variables categóricas con one-hot para poder usarlas en el modelo
X = df.drop('charges', axis=1).values
y = df['charges'].values.reshape(-1, 1)

# separamos features y objetivo, convertimos a float32 y normalizamos las columnas
# la normalización evita que las características con escala grande dominen el gradiente
y = y.astype(np.float32)

# normalizar features
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1
X = (X - X_mean) / X_std

# convertir a tensors
X = torch.tensor(X)
y = torch.tensor(y)

n_samples, n_features = X.shape

# definimos la red neuronal
# construimos una red pequeña con una capa oculta y relu para capturar no linealidades

class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# configuración de entrenamiento:
# definimos tasa de aprendizaje, número de iteraciones, función de pérdida y optimizador
# cada época calculamos predicción, pérdida, retropropagación y actualizamos pesos
model = NeuralNet(n_features)

learning_rate = 0.01
n_iters = 30000
target_loss = 19_000_000

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(n_iters):

# bucle de entrenamiento:
# cada época calculamos predicción, pérdida, retropropago y actualizamos pesos
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{n_iters}] Loss: {loss.item():.2f}')

    if loss.item() <= target_loss:
        print(f'\nTarget loss reached at epoch {epoch+1}')
        break

print("\nFinal Loss:", loss.item())