import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

# cargamos y preprocesamos los datos
# en esta sección cargamos el csv, aplicamos codificación para variables categóricas,
# separamos las características y el objetivo, convertimos los tipos y normalizamos las features.
# explicamos por qué hacemos cada paso y cómo afecta al entrenamiento.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "insurance.csv")

df = pd.read_csv(DATA_PATH)

# codificamos variables categóricas con one-hot para convertir texto en números
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# separamos las columnas en variables explicativas (x) y la variable objetivo (y)
X = df.drop('charges', axis=1).values
y = df['charges'].values.reshape(-1, 1)

# convertimos a float32 porque pytorch trabaja mejor con este tipo y ahorra memoria
X = X.astype(np.float32)
y = y.astype(np.float32)

# normalizamos las features: esto centra y escala los datos por columna,
# lo hacemos porque facilita la convergencia del optimizador y evita pasos erráticos
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

# convertimos los arrays de numpy a tensores de torch para poder entrenar el modelo
X = torch.tensor(X)
y = torch.tensor(y)

n_samples, n_features = X.shape

# definimos el modelo lineal
# aquí creamos una clase que extiende nn.Module y contiene una sola capa lineal

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression(n_features, 1)

# configuramos el entrenamiento
# definimos la tasa de aprendizaje, el número de iteraciones, el criterio y el optimizador

learning_rate = 0.001
n_iters = 30000
target_loss = 19_000_000

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# bucle de entrenamiento
# en cada iteración: calculamos la predicción, la pérdida, hacemos retropropagación y actualizamos los pesos

for epoch in range(n_iters):
    # forward: calculamos la salida del modelo sobre todo el conjunto de entrenamiento
    outputs = model(X)
    # calculamos la pérdida media cuadrática entre la predicción y el valor real
    loss = criterion(outputs, y)

    # backward: limpiamos gradientes, computamos gradientes y actualizamos parámetros
    optimizer.zero_grad()  # por seguridad, limpiamos los gradientes acumulados
    loss.backward()        # calculamos gradientes mediante backpropagation
    optimizer.step()       # actualizamos los pesos con el optimizador seleccionado

    # mostramos progreso cada 1000 épocas para seguir la convergencia sin saturar la salida
    if (epoch+1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{n_iters}], Loss: {loss.item():.4f}')

    # parada temprana si alcanzamos la pérdida objetivo para ahorrar tiempo
    if loss.item() <= target_loss:
        print(f'\nse alcanzó la loss objetivo en la época {epoch+1}')
        break

print("\nfinal loss:", loss.item())