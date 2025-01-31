# Informe sobre el Entrenamiento de un Modelo en un Grafo Heterogéneo

## 1. Introducción

El presente informe detalla el proceso de entrenamiento de un modelo basado en grafos heterogéneos, la elección de capas y arquitecturas, la estrategia de preprocesamiento de datos y la evaluación del rendimiento del modelo mediante diversas métricas y técnicas de validación cruzada.

## 2. Elección del Grafo Heterogéneo

Se optó por entrenar sobre un **grafo heterogéneo** debido a su capacidad para modelar de manera más realista la estructura de los datos. Aunque estos grafos presentan mayor complejidad en su manejo, ofrecen la ventaja de permitir la aplicación de distintas capas a cada tipo de nodo en caso de ser necesario en futuras optimizaciones.

Dentro de nuestro grafo, se implementó una **capa lineal** encargada de transformar el vector de características de los nodos (ligandos y proteínas) en un vector de tamaño 5 para las capas ocultas. A continuación, se utilizaron **cuatro capas HeteroConvolucionales**, encargadas de mezclar (mediante suma) las salidas obtenidas tras aplicar **SAGEConv** a ambos tipos de nodos.

El mecanismo de **SAGEConv** realiza la agregación de características en cada nodo utilizando la media de los valores de sus vecinos. Finalmente, se añadió una última **capa lineal** desde la cual se extraen los valores finales de los ligandos.

### Comparación de Capas GNN

Se evaluaron distintas opciones para el diseño de la arquitectura, optando finalmente por **HeteroConv** debido a sus ventajas en el tratamiento de grafos heterogéneos. A continuación, se presenta un resumen de las capas consideradas:

| **Capa**       | **Descripción**                                                                        | **Cuándo usarla**                                                                                   |
| -------------- | -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **HeteroConv** | Aplica diferentes capas GNN a cada relación y las combina.                             | Si se requiere definir una función de propagación distinta por tipo de relación.                    |
| **RGCNConv**   | Usa una única matriz de pesos compartida para manejar múltiples relaciones.            | Si hay muchas relaciones diferentes y se necesita un modelo más eficiente.                          |
| **HANConv**    | Usa atención jerárquica para aprender qué relaciones y conexiones son más importantes. | Si se trabaja con múltiples relaciones y se desea que la GNN aprenda cuáles son las más relevantes. |

Dado que nuestro problema se beneficia de la posibilidad de aplicar diferentes propagaciones a cada tipo de relación, **HeteroConv** fue la opción seleccionada.

---

## 3. Preprocesamiento de Datos

Para la preparación de los datos se utilizó **one-hot encoding** en la identificación de características como **htype, irr, dssp y mol**.

Se consideraron además futuras mejoras en el preprocesamiento, como:

- **Normalización de posiciones (x, y, z)** en un rango de **-1 a 1**, para mejorar la estabilidad del modelo.
- **Uso de embeddings posicionales** para capturar de manera más efectiva la estructura tridimensional de los datos.

---

## 4. Evaluación del Modelo

Para medir el desempeño del modelo, se utilizaron tres métricas principales:

1. **MSE (Error Cuadrático Medio):** Evalúa la precisión del modelo en la predicción de los valores de salida.
2. **MAE (Error Absoluto Medio):** Proporciona una medida directa de la desviación media en las predicciones.
3. **R² (Coeficiente de Determinación):** Indica qué tan bien explica el modelo la variabilidad de los datos.

Además, se aplicó **K-Fold Cross-Validation** para evaluar la robustez de los modelos que mejores resultados iban dando.

### K-Fold Cross-Validation

La **validación cruzada de K pliegues** es una técnica que permite evaluar la capacidad de generalización de un modelo. En lugar de dividir el conjunto de datos en solo dos partes (entrenamiento y prueba), se divide en **K subconjuntos** (o “folds”) de tamaño similar. Luego, el modelo se entrena y evalúa **K veces**, utilizando cada fold una vez como conjunto de validación y los **K−1 folds restantes** como conjunto de entrenamiento.

#### Ventajas de K-Fold Cross-Validation

- **Mejor estimación del rendimiento:** Se obtiene una estimación más robusta y confiable del desempeño del modelo en comparación con una sola división de datos.
- **Uso eficiente de los datos:** Cada dato se emplea tanto para entrenamiento como para validación, lo que es especialmente útil en conjuntos de datos pequeños.
- **Detección de sobreajuste:** Si el modelo muestra una variabilidad significativa en los resultados de cada fold, podría estar sobreajustando a los datos.

---

## 5. Experimentación con Funciones de Activación

Se probaron diferentes funciones de activación para optimizar el desempeño del modelo, incluyendo:

- **ReLU**
- **ELU**
- **Swish**

Inicialmente, la función **Swish** mostró los mejores resultados en las primeras 50 épocas. Sin embargo, se observó que, a pesar de una mejora continua en la función de pérdida, la validación permanecía estancada. Esto sugiere que el modelo estaba **aprendiendo bien los datos de entrenamiento pero sin generalizar adecuadamente**.

Para abordar este problema, se tomaron las siguientes medidas:

1. **Aumento del número de neuronas** en las capas ocultas a **100** para mejorar la capacidad de generalización.
2. **Incremento del número de épocas a 100**, ya que la curva de aprendizaje, aunque errática, mostraba potencial de convergencia.

Sin embargo, estos cambios **no dieron resultados significativamente mejores**, lo que sugiere la necesidad de ajustar otros parámetros, como la regularización o la arquitectura del modelo.

---

## 6. Conclusión y Futuras Mejoras

El entrenamiento en un grafo heterogéneo permitió modelar de manera más realista las interacciones en los datos, y la elección de **HeteroConv** resultó adecuada para nuestro problema. Se realizaron diversos ajustes en la arquitectura y en las funciones de activación para optimizar el desempeño del modelo.

**Posibles mejoras futuras incluyen:**

- **Normalización de las posiciones espaciales** para mejorar la estabilidad del entrenamiento.
- **Uso de embeddings posicionales** para capturar mejor la estructura tridimensional de los datos.
- **Ajuste de hiperparámetros** adicionales, como regularización y dropout, para mejorar la generalización.
- **Experimentación con más arquitecturas de GNN**, como **GATConv o GraphSAGE**, para evaluar posibles mejoras en la propagación de información.

A pesar de los desafíos encontrados en la validación del modelo, las pruebas realizadas proporcionan un punto de partida sólido para futuras optimizaciones.
