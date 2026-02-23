
# 1.11 Regression Model – Medical Cost Prediction

## Description

This project implements a Linear Regression model to predict medical insurance charges using the Medical Cost Personal Dataset.

The goal is to achieve:

Loss ≤ 19,000,000

This is a team activity.

---

## Team Members

Rodrigo Martin del Campo Arroyo  
Jose Juan Diaz Campos  
Sergio Villa Rodriguez  
Mateo Garcia Lopez  

---

## Descripción del desarrollo

Para este proyecto se implementaron dos enfoques diferentes para predecir la variable `charges`:

1. Regresión Lineal
2. Red Neuronal Simple (alternativa para alcanzar el loss solicitado por el profesor)

Antes del entrenamiento se realizaron los siguientes pasos de preprocesamiento:

- Codificación One-Hot para las variables categóricas (`sex`, `smoker`, `region`).
- Normalización de las características (X) utilizando estandarización.
- Normalización de la variable objetivo (`charges`) para mejorar la estabilidad del entrenamiento.

La normalización fue necesaria debido a que los valores de `charges` pueden alcanzar hasta aproximadamente 60,000, lo que genera valores muy grandes en el cálculo del MSE.

---

## Resultados con Regresión Lineal

Se implementó un modelo de regresión lineal utilizando una sola capa totalmente conectada.

A pesar de aplicar normalización y utilizar el optimizador Adam para mejorar la convergencia, los resultados no fueron los esperados.

El loss (MSE en escala original) se estabilizó aproximadamente en:

~35,000,000

Incluso en epochs avanzados (cercanos a 30,000), el valor del loss dejó de disminuir y presentó un comportamiento de meseta.

Esto indica que la relación entre las variables del dataset y `charges` no es completamente lineal, por lo que el modelo no tiene suficiente capacidad para capturar los patrones presentes en los datos.

---

## Resultados con Red Neuronal

Posteriormente se implementó una red neuronal simple con:

- Una capa oculta
- Función de activación ReLU
- Optimizador Adam

La arquitectura permitió capturar relaciones no lineales entre variables como:

- La interacción entre `smoker` y `charges`
- El efecto combinado de `age` y `bmi`
- Otros patrones no lineales presentes en el dataset

Con esta implementación, el modelo sí logró alcanzar el objetivo:

Loss ≤ 19,000,000

La red neuronal mostró una mejor capacidad de aprendizaje y una convergencia más estable en comparación con la regresión lineal.

---

## Conclusión

La regresión lineal no fue suficiente para alcanzar el loss objetivo, ya que se estabilizó alrededor de 35,000,000 incluso después de un número elevado de epochs.

En cambio, la red neuronal simple logró capturar patrones más complejos en los datos y alcanzar el loss requerido.

Este resultado demuestra la importancia de seleccionar un modelo con suficiente capacidad para representar la complejidad del problema.
