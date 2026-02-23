import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Cargar datos preprocesados
X = pd.read_csv('data/X_scaled.csv')
y = pd.read_csv('data/y.csv').squeeze()

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=25)

# Entrenamiento del modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Generación de predicciones
y_pred = model.predict(X_test)

# Exportar métricas para evaluación
results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})
results.to_csv("reports/results.csv", index=False)