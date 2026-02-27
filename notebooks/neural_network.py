import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt 

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
# convertimos a float32 para que numpy/pytorch opere correctamente
X = X.astype(np.float32)
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

historial_loss = [] 
for epoch in range(n_iters):

# bucle de entrenamiento:
# cada época calculamos predicción, pérdida, retropropago y actualizamos pesos
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    historial_loss.append(loss.item()) 

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{n_iters}] Loss: {loss.item():.2f}')

    if loss.item() <= target_loss:
        print(f'\nTarget loss reached at epoch {epoch+1}')
        break

print("\nFinal Loss:", loss.item())

# guardar métricas y predicciones en reports/
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

y_pred = outputs.detach().cpu().numpy().ravel()
y_true = y.detach().cpu().numpy().ravel()

final_mse = metrics.mean_squared_error(y_true, y_pred)
final_mae = metrics.mean_absolute_error(y_true, y_pred)
final_rmse = np.sqrt(final_mse)
final_r2 = metrics.r2_score(y_true, y_pred)

metrics_df = pd.DataFrame({
    'metric': ['final_loss', 'mse', 'mae', 'rmse', 'r2'],
    'value': [loss.item(), final_mse, final_mae, final_rmse, final_r2]
})
metrics_df.to_csv(os.path.join(REPORTS_DIR, 'neural_metrics.csv'), index=False)

pd.DataFrame({'Actual': y_true, 'Predicted': y_pred}).to_csv(os.path.join(REPORTS_DIR, 'neural_evaluation_results.csv'), index=False)

print('saved neural metrics to', os.path.join(REPORTS_DIR, 'neural_metrics.csv'))
print('saved neural evaluation results to', os.path.join(REPORTS_DIR, 'neural_evaluation_results.csv'))

plt.figure(figsize=(10, 6))
plt.plot(historial_loss, color='blue', label='Error de la Red Neuronal')
plt.axhline(y=19000000, color='red', linestyle='--', label='Meta: 19 Millones')
plt.title('Descenso del Error (Loss) - Red Neuronal')
plt.xlabel('Épocas')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(BASE_DIR, 'reports', 'curva_aprendizaje_nn.png')) # Guarda la imagen
plt.show()