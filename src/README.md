# Código de DeepGProp

Este directorio está compuesto por distíntos módulos usados para obtener
resultados comentados en la documentación.

Se distribuyen de la siguiente manera:

- [`dgp_logger.py`]: registrador de mensajes para uso en los distintos
  módulos.
- [`deep_g_prop.py`]: interfaz de línea de comandos para lanzar una ejecución
  de algoritmos genéticos sobre un modelo de perceptrón multicapa.
- [`keras_model_creator.py`]: interfaz de línea de comandos para crear modelos de
  Keras y poder utilizarlos con `deep_g_prop.py`.
- [`ga_optimizer.py`]: implementación de un algoritmo genético.
- [`types.py`]: distintos tipos de datos utilizados en varios módulos.
- [`utils.py`]: funciones útiles para diversos casos.
- [`common.py`]: variables comunes utilizadas por distintos módulos.


- [`models`]: carpeta donde se pueden encontrar los modelos creados con
  `keras_model_creator.py`.
- [`datasets`]: carpeta en la que se incluyen los conjuntos de datos utilizados
  en el código.
- [`hp_optimization`]: conjuntos de scripts utilizados para probar distintos
  optimizadores de hiperparámetros.
- [`mlp_frameworks`]: conjunto de scripts utilizados para probar distintos
  frameworks y elegir uno para el trabajo.

<!-- URLs -->
[`dgp_logger.py`]: ./dgp_logger.py
[`deep_g_prop.py`]: ./deep_g_prop.py
[`keras_model_creator.py`]: ./keras_model_creator.py
[`ga_optimizer.py`]: ./ga_optimizer.py
[`types.py`]: ./types.py
[`utils.py`]: ./utils.py
[`common.py`]: ./common.py
[`models`]: ./models
[`datasets`]: ./datasets
[`hp_optimization`]: ./hp_optimization
[`mlp_frameworks`]: ./mlp_frameworks