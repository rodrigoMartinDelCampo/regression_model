import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/insurance.csv')

print("--- Info ---")
df.info()
print("\n--- Stats ---")
print(df.describe())
print("\n--- Nulls ---")
print(df.isnull().sum())

# Encoding categóricas
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Separar features y target
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']

# Scaling de features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Exportar datasets limpios
X_scaled.to_csv("data/X_scaled.csv", index=False)
y.to_csv("data/y.csv", index=False)