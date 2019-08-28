# Trabajo de Fin de Grado: *DeepGProp*

> Optimización de Redes Neuronales con Algoritmos Genéticos

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

Primero se debe de tener python 3.7 y pip instalado en el sistema. Después
instala el paquete `pipenv` para gestión de dependencias y entornos virtuales.
Para ello se puede utilizar pip:

```bash
pip install pipenv
```

Una vez instalado el paquete, ejecutando la siguiente línea se podrán utilizar
las funciónes principales del código:

```bash
pipenv install
```

Si se quieren ejecutar los tests se pueden instalar los paquetes de desarrollo:

```bash
pipenv install --dev
```

Además se deben instalar los paquetes `pandoc`, `aspell` y `aspell-es` con el
gestor de paquetes que se prefiera.

## Utilización

## Desarrollo

En este proyecto se usa un metodología de desarrollo basada en tests.

Se utilizará también un fichero de log para ir destacando el trabajo que se
realiza en el proyecto y otro para guardar todas las pruebas que se realizen con
código. Éstos tendrán una estructura parecida al ["ChangeLog" de
GNU](https://www.gnu.org/software/emacs/manual/html_node/emacs/Format-of-ChangeLog.html).

## Frameworks

-   [Keras](https://keras.io/) - librería para la creación y ejecución de redes
    neuronales. Se ha elegido ésta frente a otras librerías por su sensillez de
    uso, buena documentación y soporte (dada la gran comunidad que la utiliza).

-   [DEAP](https://deap.readthedocs.io/en/master/) - librería de construcción
    de algoritmos evolutivos. Se utilizará ésta para optimizar los parámetros
    de las redes neuronales.

## Utilidades

-   General:

    -   [Sultan](https://sultan.readthedocs.io/en/latest/) - librería en python
        para ejecutar processos de manera cómoda.

-   Automatización:

    -   [Invoke](http://docs.pyinvoke.org/en/1.2/) - utilidad para ejecutar
        procesos como la construcción de la documentación.

-   Tests:

    -   [pytest](https://docs.pytest.org/en/latest/) - librería de python para
        la ejecución de tests que se usarán en la integración continua.

    -   [pandoc](https://pandoc.org/MANUAL.html) y
        [aspell](http://aspell.net/man-html/Introduction.html#Introduction) -
        utilidades para convertir la documentación del trabajo a texto plano y
        comprobar si hay errores ortográficos.

    -   [PyPDF](http://mstamy2.github.io/PyPDF2/) - otra utilidad para comprobar
        distintas características del PDF resultante de documentación.

-   Documentación:

    -   [Pweave](http://mpastell.com/pweave/) - módulo de python que permite
        mostrar salida de código directamente en LaTeX.

    -   [TexLive](https://tug.org/texlive/) - generador de archivos PDFs.

## Licencia

El código de este repositorio está liberado bajo la licencia [GPL](./LICENSE).
