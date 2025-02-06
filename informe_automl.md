# **Informe de AutoML para la predicción de valores STD de proteínas con ligandos**

## **1. Introducción**
Este proyecto utiliza **Auto-sklearn**, una herramienta de Auto Machine Learning (AutoML), para predecir los valores de **STD (007_score_ligando)** en una base de datos que contiene información sobre **proteínas y ligandos**. El objetivo es encontrar el mejor modelo para realizar predicciones precisas a partir de las características extraídas.

Se han realizado varias configuraciones de entrenamiento con distintos **tiempos de entrenamiento**, **tamaños de ensamble**, **estrategias de validación** y **manejo de valores faltantes**, con el fin de optimizar el rendimiento del modelo.

---

## **2. Configuración de los experimentos**
Se realizaron **siete experimentos** con distintas configuraciones en Auto-sklearn. Las diferencias principales entre cada experimento fueron:

| **Experimento** | **Tiempo de entrenamiento** | **Tiempo por Modelo** | **Estrategia de validación** | **Ensamble** | **Manejo de valores faltantes** |
|---------------|-------------------------------|------------------------|---------------------------|-------------|------------------|
| 1 | 300 | 30 | Holdout (default) | 50 | AutoML imputación (media) |
| 2 | 600 | 60 | Holdout | 50 | AutoML imputación (media) |
| 3 | 1800 | 90 | Holdout | 100 | AutoML imputación (media) |
| 4 | 300 | 30 | **Cross Validation (5 folds)** | 50 | AutoML imputación (media) |
| 5 | 300 | 30 | Holdout | 50 | Eliminación de columnas con muchos valores faltantes |
| 6 | 300 | 30 | Holdout | 50 | Relleno con valor extremo (`min_val - 9964.995`) |
| 7 | 300 | 30 | Holdout | **20 (menor ensamble)** | **Sin metalearning** |

---

## **3. Resultados obtenidos**
A continuación, se presentan los resultados de cada experimento en términos de **Mean Squared Error (MSE)**, **Mean Absolute Error (MAE)** y **R² Score**:

| **Experimento** | **MSE (↓ mejor)** | **MAE (↓ mejor)** | **R² Score (↑ mejor)** | **Mejor modelo** | **Observaciones** |
|---------------|------------|------------|----------|---------------|------------------|
| 1 | **0.0143** | 0.0920 | **0.7786** | **Gradient Boosting (98%) + Extra Trees (2%)** | Solo encuentra **Gradient Boosting** |
| 2 | **0.0135** | **0.0900** | **0.7903** | **Gradient Boosting** | Sigue favoreciendo **Gradient Boosting** |
| 3 | **0.0133** | **0.0893** | **0.7939** | **Gradient Boosting** | Aumentar tiempo mejora el rendimiento ligeramente |
| 4 | **0.0310** | **0.1411** | **0.5198** | **Linear SVR** | **Cross Validation empeoró el rendimiento** |
| 5 | **0.0139** | **0.0911** | **0.7841** | **Gradient Boosting** | Eliminación de columnas no impactó significativamente |
| 6 | **0.0139** | **0.0914** | **0.7840** | **Gradient Boosting** | Uso de valor extremo para faltantes no afectó mucho |
| 7 | **0.0209** | **0.1139** | **0.6766** | **KNN (55%) + Decision Tree (40%) + ARD Regression (5%)** | **Sin metalearning favorece más modelos** |

**Mejor experimento:** `#3` (MSE más bajo, R² más alto).  
**Peor experimento:** `#4` (Cross Validation redujo el desempeño drásticamente).  
**Efecto de la eliminación de valores faltantes:** No impactó mucho los resultados.  
**Efecto de la eliminación de metalearning (`#7`)**: Permitió encontrar otros modelos, pero con menor rendimiento.  

---

## **4. Análisis de Resultados**
### **4.1. El dominio de Gradient Boosting**
En la mayoría de los experimentos, **Gradient Boosting** (HistGradientBoostingRegressor) fue el modelo dominante, lo que indica que **es la mejor opción para estos datos**. Otras técnicas como **KNN y Decision Trees** solo aparecen cuando **se desactiva el metalearning (`#7`)**.

### **4.2. Cross Validation no mejoró el modelo**
El experimento `#4` usó **Cross Validation (5 folds)**, pero esto **redujo el rendimiento considerablemente**. Esto sugiere que **la partición original de train/test era suficiente y que la validación cruzada no aporta valor en este caso**.

### **4.3. Manejo de valores faltantes**
Probar diferentes estrategias de manejo de valores faltantes (`#5` y `#6`) **no tuvo un gran impacto** en los resultados, lo que indica que **el método de imputación de AutoML ya estaba funcionando bien**.


---

## **5. Recomendaciones**
**Optimización manual de hiperparámetros de Gradient Boosting** para mejorar aún más el rendimiento.  
**Explorar ajustes en el ensamble** para diversificar los modelos usados.  
**Probar técnicas de selección de características** para reducir la dimensionalidad.  