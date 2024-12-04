import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# 1. Cargar y preparar los datos
# Cargar el archivo CSV
data = pd.read_csv('owid-energy-data.csv')

# Filtrar los datos para Argentina
argentina_data = data[data['country'] == 'Argentina']

# Eliminar valores faltantes en el consumo energético
argentina_data = argentina_data.dropna(subset=['primary_energy_consumption'])

# Variables independientes (años) y dependientes (consumo energético)
X = argentina_data[['year']]  # Año como característica
y = argentina_data['primary_energy_consumption']  # Consumo energético como objetivo

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Crear y ajustar un modelo polinómico de grado 2
# Crear características polinómicas de grado 2
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)  # Transformar características de entrenamiento
X_test_poly = poly.transform(X_test)  # Transformar características de prueba

# Entrenar un modelo de regresión lineal sobre las características polinómicas
model = LinearRegression()
model.fit(X_train_poly, y_train)

# 3. Evaluar el modelo
# Hacer predicciones sobre los datos de prueba
y_pred = model.predict(X_test_poly)

# Calcular métricas de evaluación: MAE y R²
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE (Polinómico grado 2): {mae:.2f}, R² (Polinómico grado 2): {r2:.2f}")

# 4. Predicciones futuras (2025-2035)
# Crear un rango de años futuros para extrapolar el modelo
future_years = pd.DataFrame({'year': range(2025, 2035)})  # Usar DataFrame para mantener consistencia
future_years_poly = poly.transform(future_years)

# Predecir el consumo energético para los años futuros
future_predictions = model.predict(future_years_poly)

# 5. Visualización mejorada
# Gráfica de datos históricos con seaborn
plt.figure(figsize=(12, 6))
sns.scatterplot(x='year', y='primary_energy_consumption', data=argentina_data, color='blue', alpha=0.7, label='Datos Históricos')
sns.lineplot(x='year', y=model.predict(poly.transform(X)), data=argentina_data, color='red', label='Ajuste Polinómico (Grado 2)')
plt.title('Consumo Energético en Argentina: Ajuste Polinómico Grado 2', fontsize=16)
plt.xlabel('Año', fontsize=12)
plt.ylabel('Consumo Energético (TWh)', fontsize=12)
plt.legend()
plt.grid()
plt.show()

# Grafica interactiva con Plotly
fig = px.scatter(argentina_data, x='year', y='primary_energy_consumption', 
                 labels={'year': 'Año', 'primary_energy_consumption': 'Consumo Energético (TWh)'}, 
                 title='Consumo Energético en Argentina: Datos Históricos y Predicciones')
fig.add_scatter(x=argentina_data['year'], y=model.predict(poly.transform(X)), mode='lines', name='Ajuste Polinómico (Grado 2)', line=dict(color='red'))
fig.add_scatter(x=future_years['year'], y=future_predictions, mode='lines+markers', name='Predicciones Futuras', line=dict(dash='dash', color='green'))
fig.update_layout(template='plotly_white')
fig.show()

# Grafica interactiva de residuales con matplotlib
residuals = y_test - y_pred
plt.figure(figsize=(10, 5))
sns.residplot(x=y_pred, y=residuals, lowess=True, color='purple', scatter_kws={'alpha': 0.6})
plt.axhline(0, color='black', linestyle='--', linewidth=1)  
plt.title('Gráfica de Residuales', fontsize=16)
plt.xlabel('Valores Predichos', fontsize=12)
plt.ylabel('Residuales (Errores)', fontsize=12)
plt.grid()
plt.show()
