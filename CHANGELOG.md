# Changelog

Todos los cambios importantes de este proyecto se mostrarán aquí.

El formato se basa en
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/), y este proyecto sigue
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Cambios inéditos]

### Añadido

- Añadidos operadores de agregar/quitar neuronas/capas.

- Nueva herramienta para obtener conjuntos de datos tipo Proben1 a partir de
  problemas tipo Spambase (todo incluido en un csv).

### Modificado

- Refactorizado módulo `ga_optimizer`. Ahora admite como entrada una secuencia
  que transformará en configuraciones de capas.

- Arreglado problema con la función de fintess y el cruce de individuos.

### Eliminado

- Eliminado el módulo de creación de modelos para Keras, ya que con la nueva
  configuración del algoritmo genético, no es necesario.

## [0.0.3] - 2020-06-19

### Añadido

- Añadido algoritmo genético básico para evolucionar configuraciónes de
  perceptrones multicapa.

- Revisado README y añadida información para la utilización de la nueva
  herramienta de automatización: Nox.

- Módulo para crear modelos de keras y usarlos como entrada del algoritmo
  genético.

### Modificado

- Cambio de proveedor para integración contínua: de Travis CI a GitHub Actions.

- Cambio de proveedor de automatización: de Invoke a Nox.

- Mejorado el registrador de datos `dgp_logger`.

### Eliminado

- Automatización con Invoke.

## [0.0.2] - 2020-06-10

### Añadido

- Comparativa de distintos optimizadores de hyper-parámetros.

- Tests para todo el código existente.

- Explicación de desarrollo con venv.

- Comparativa entre distintos frameworks para redes neuronales artificiales.

- Tests para este nuevo código.

### Modificado

- Cambio de proveedor de entorno virtual a venv.

### Eliminado

- Entorno virtual basado en Pipenv.

## [0.0.1] - 2020-03-22

### Añadido

- Añadido archivo de  licencia.

- Añadida portada, Introducción y Descripción del Problema, y resumen en
  Español.

- Añadidos algunos test para la documentación y revisión ortográfica.

- Añadidas tareas básicas para construir la documentación, limpiar el
  repositorio y ejecutar test.

- Añadida Integración Continua con Travis CI.

- Añadida información en el archivo README.md sobre la instalación, uso,
  desarrollo, frameworks y utilidades del proyecto.

- Añadido Pipfile para controlar los paquetes de Python.

[Cambios inéditos]: https://github.com/lulivi/deep-g-prop/compare/v0.0.3...HEAD
[0.0.1]: https://github.com/lulivi/deep-g-prop/releases/tag/v0.0.1
[0.0.2]: https://github.com/lulivi/deep-g-prop/releases/tag/v0.0.2
[0.0.3]: https://github.com/lulivi/deep-g-prop/releases/tag/v0.0.3
