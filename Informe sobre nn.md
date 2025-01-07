# Informe Visual: Conceptos Clave en Machine Learning



## 1. Exploración de Datos Inicial

**Visuales a Incluir:**

- Un par de gráficos de distribución de datos (p. ej., histogramas o gráficos de dispersión).

**Tareas Principales:**

- Determinar si los datos siguen una distribución normal.
- Representar visualmente los datos.

---

## 2. Funciones de Pérdida a Considerar

**Diagrama Comparativo:**

- **MSE (Error Cuadrático Medio):** Breve explicación con ecuación.
- **Log(cosh(error)):** Breve explicación con ecuación.

**Visuales a Incluir:**

- Un gráfico comparativo para ver cómo reaccionan estas funciones a diferentes errores.

---

## 3. Funciones de Activación

**Visualización de Funciones de Activación:**

- Enlace a Datacamp o un resumen gráfico de las funciones de activación populares, como ReLU, Sigmoid, Tanh, etc.

**Visuales a Incluir:**

- Gráficos que muestren la forma de cada función de activación y cómo responde en diferentes escenarios (input vs output).

---

## 4. Optimización: Métodos de Descenso de Gradiente

**Tabla Resumida (Tipo Infografía):**

- Comparación de BGD, SGD, MBGD, Adam, RMSprop y Adagrad.

**Visuales a Incluir:**

- **Tabla Visual:**
  - **Método** (con iconos representativos).
  - **Actualización de Pesos**.
  - **Ventajas**.
  - **Desventajas**.
  - **Uso Recomendado**.
- **Visual Extra:**
  - Un gráfico de línea que muestre una comparación ficticia de la convergencia de estos métodos a través de iteraciones.

---

## 5. Métricas de Evaluación

**Definiciones con Ejemplos Visuales:**

- **Métricas:** Accuracy, Precision, Recall, F1-Score.

**Visuales a Incluir:**

- Diagrama tipo matriz para ilustrar la diferencia entre las métricas, con diagramas de matrices de confusión y cómo afectan cada métrica.

---

## 6. Métricas Finales de Rendimiento

**Comparación de MAE, MSE, R²:**

- **MAE (Error Absoluto Medio):** Explicación con ejemplos gráficos de errores absolutos.
- **MSE (Error Cuadrático Medio):** Comparación de cómo pondera los errores grandes.
- **R² (Coeficiente de Determinación):** Visual que muestra cómo de bien se ajusta el modelo a los datos.

**Visuales a Incluir:**

- Gráfico que compare la diferencia entre el MAE, MSE y R² para un conjunto de datos predicho vs observado.

---

## 7. Errores Numéricos en PyTorch y Estrategias para Manejarlos

**Bloque Dividido en Dos Partes:**

1. **Lo Que Hace PyTorch:**

   - Inicialización automática, herramientas para clipping, Batch Normalization, etc.
   - **Visuales a Incluir:** Diagrama de flujo sobre el proceso de manejo automático de errores por PyTorch.

2. **Lo Que Tú Debes Hacer:**

   - Preprocesamiento de datos, personalización de pesos, BatchNorm.
   - **Visuales a Incluir:** Diagrama que muestre los pasos de preprocesamiento.

---

## 8. Inicialización de Pesos

**Tipos de Inicialización:**

- **Métodos:** Xavier, He, Uniforme.
- **Descripción:** Explicación breve de cada método y cuándo se usa.

**Visuales a Incluir:**

- Gráfico que muestre las distribuciones iniciales de los pesos generados por cada método.

---

## 9. Batch Normalization

**Tipos y Ventajas:**

- **BatchNorm para Capas Densas vs Convolucionales:**
  - Diferencias clave, cuándo usar cada uno y sus ventajas.

**Visuales a Incluir:**

- Infografía simple que muestre los beneficios de implementar BatchNorm en una red profunda.

---

## 10. Clipping de Gradientes

**Definición y Cuándo Usar:**

- **Descripción:** ¿Qué es el clipping? Escenarios donde los gradientes explotan (problemas en redes profundas o RNNs).

**Visuales a Incluir:**

- Gráfico que muestre un ejemplo de gradientes que se vuelven inestables y cómo el clipping limita su magnitud.
