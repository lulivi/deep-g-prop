# Trabajo de Fin de Grado: *DeepGProp*

> Optimización de Redes Neuronales con Algoritmos Genéticos

[![Build Status](https://travis-ci.org/lulivi/deep-g-prop.svg?branch=master)](https://travis-ci.org/lulivi/deep-g-prop)

**Autor(a): Luis Liñán Villafranca**

**Tutor(a)(es): Juan Julián Merelo Guervós**

---

- [Instalación](#instalación)
- [Utilización](#utilización)
- [Desarrollo](#desarrollo)
- [Frameworks](#frameworks)
- [Utilidades](#utilidades)
- [Licencia](#licencia)

---

## Instalación

Primero se debe de tener Python 3.7 y `pip` instalados en el sistema. Después
se podrá elegir si utilizar el entorno virtual que viene con el repositorio, o
instalar los paquetes manualmente.

1.  Si se elige la primera opción, se deberá instalar el paquete `pipenv` para
    la gestión de dependencias y entornos virtuales. Para ello se puede
    utilizar `pip`:

    ```shell
    pip install pipenv
    ```

    Una vez instalado el paquete, ejecutando la siguiente línea se podrán utilizar
    las funciónes principales del código:

    ```shell
    pipenv install
    ```

2.  Por otro lado, si se quieren instalar los módulos de Python en el sistema
    sin utilizar un entorno virtual se podrá utilizar `pip`:

    ```shell
    python3.7 -m pip install --user <paquete> [<paquete> ...]
    ```

    Siendo los paquetes los contenidos en el fichero [`Pipfile`](./Pipfile):

    -   Para instalar los paquetes mínimos para ejecutar el proyecto, tendrá
        que escogerlos de la etiqueta `[packages]`.

    -   Si también quiere instalar los paquetes para tests, deberá añadirles
        los que se encuentran bajo la etiqueta `[dev-packages]`.

    Además se deben instalar los programas [pandoc][pandoc], [aspell][aspell] y
    el [diccionario de español][aspell-es] para aspell (aspell-es en Linux).

## Utilización

> **Nota:** Si se ha optado por utilizar el gestor de entornos virtuales
> `pipenv`, primero deberá cargar el entorno ejecutando lo siguiente:
>
> ```shell
> pipenv shell
> ```
>
> Si no quiere, puede ejecutar cualquiera de los siguientes comandos
> precediendo siempre con:
>
> ```shell
> pipenv run <comando>
> ```

Para poder construir la documentación se puede ejecutar lo siguiente:

```shell
inv docs.pdf
```

Para ejecutar los tests:

```shell
inv tests.unittest
```

Para obtener otras opciones posibles con `invoke`:

```shell
inv -l
```

## Desarrollo

En este proyecto se usa un metodología de desarrollo basada en tests.

Se utilizará también un fichero de log para ir destacando el trabajo que se
realiza en el proyecto y otro para guardar todas las pruebas que se realicen con
código. Éstos tendrán una estructura parecida al ["ChangeLog" de
GNU][changelog].

## Frameworks

-   [Keras][keras] - librería para la creación y ejecución de redes
    neuronales. Se ha elegido ésta frente a otras librerías por su sensillez de
    uso, buena documentación y soporte (dada la gran comunidad que la utiliza).

-   [DEAP][deap] - librería de construcción
    de algoritmos evolutivos. Se utilizará ésta para optimizar los parámetros
    de las redes neuronales.

En la documentación se podrá encontrar una comparativa detallada con otras
bibliotecas similares y el por qué de la elección de éstas.

## Utilidades

-   General:

    - [Sultan][sultan] - librería en python para ejecutar processos de manera
      cómoda.

-   Automatización:

    - [Invoke][invoke] - utilidad para ejecutar procesos como la construcción de
      la documentación.

-   Tests:

    - [pytest][pytest] - librería de python para la ejecución de tests que se
      usarán en la integración continua.

    - [pandoc][pandoc] y [aspell][aspell] - utilidades para convertir la
      documentación del trabajo a texto plano y comprobar si hay errores
      ortográficos.

    - [PyPDF][PyPDF] - otra utilidad para comprobar distintas características
      del PDF resultante de documentación.

-   Documentación:

    - [Pweave][pweave] - módulo de python que permite mostrar salida de código
      directamente en LaTeX.

    - [TexLive][texlive] - generador de archivos PDFs a partir de Latex.

## Licencia

El código de este repositorio está liberado bajo la licencia [GPL](./LICENSE).

[pandoc]: https://pandoc.org/
[aspell]: http://aspell.net/
[aspell-es]: https://ftp.gnu.org/gnu/aspell/dict/es/
[changelog]:
https://www.gnu.org/software/emacs/manual/html_node/emacs/Format-of-ChangeLog.html
[keras]: https://keras.io/
[deap]: https://deap.readthedocs.io/en/master/
[sultan]: https://sultan.readthedocs.io/en/latest/
[invoke]: http://docs.pyinvoke.org/en/1.2/
[pytest]: https://docs.pytest.org/en/latest/
[aspell]: http://aspell.net/man-html/Introduction.html#Introduction
[PyPDF]: http://mstamy2.github.io/PyPDF2/
[pweave]: http://mpastell.com/pweave/
[texlive]: https://tug.org/texlive/