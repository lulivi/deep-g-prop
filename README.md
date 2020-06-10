# Trabajo de Fin de Grado: *DeepGProp*

> Optimización de Redes Neuronales con Algoritmos Genéticos

[![travis-badge]][travis-url]
[![license-badge]][`LICENSE`]
[![tag-badge]][`CHANGELOG.md`]

**Autor(a): Luis Liñán Villafranca**

**Tutor(a)(es): Juan Julián Merelo Guervós**

**Índice**
- [Instalación](#instalación)
- [Utilización](#utilización)
- [Desarrollo](#desarrollo)
- [Frameworks](#frameworks)
- [Utilidades](#utilidades)
- [Licencia](#licencia)

---

## Instalación

Como primer requisito, se debe de tener [Python 3.7][python-downloads-url] y
[pip] instalados en el sistema. Es muy recomendable crear un
entorno virtual para aislar correctamente las versiones de los paquetes que se
vayan a utilizar. Para más información sobre [pip] y [venv] consultar el
[tutorial oficial][python-venv-pip-guide-url].

- Para crear un entorno virtual, podemos usar el módulo que viene incorporado
  con la instalación de Python desde la versión `3.3`:

  ```shell
  python3.7 -m venv .venv
  ```

  Así habríamos creado un entorno virtual en el directorio `.venv`. Una vez
  instalado el entorno virtual, deberemos activarlo. Para ello hay que ejecutar
  uno de los siguientes comandos dependiendo del interprete de órdenes que se
  use (tabla obtenida de la documentación oficial de [venv]):

  <a name="table1.1"></a>

  | Platform |      Shell      | Command to activate virtual environment |
  | :------: | --------------: | --------------------------------------- |
  |  POSIX   |        bash/zsh | `$ source <venv>/bin/activate`          |
  |          |            fish | `$ . <venv>/bin/activate.fish`          |
  |          |        csh/tcsh | `$ source <venv>/bin/activate.csh`      |
  |          | PowerShell Core | `$ <venv>/bin/Activate.ps1`             |
  | Windows  |         cmd.exe | `C:\> <venv>\Scripts\activate.bat`      |
  |          |      PowerShell | `PS C:\> <venv>\Scripts\Activate.ps1`   |

  Tabla 1.1: *Activación de entorno virtual.*

- Para instalar los paquetes una vez activado el entorno virtual, deberemos
  usar [pip]. He dividido los paquetes en distintos grupos:

  | Propósito                          | Ruta del archivo                     | Descripción                                                                                                             |
  |------------------------------------|--------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
  | Producción                         | [`requirements/prod.txt`]            | Paquetes necesarios para ejecutar código asociado directamente a *DeepGProp*. El archivo [`requirements.txt`] de la raiz del repositorio símplemente instala los paquetes contenidos en este archivo. |
  | Documentación                      | [`requirements/docs.txt`]            | Paquetes necesarios para construir la documentación. Si se usa [Nox], no será necesario instalar estos paquetes a mano. |
  | Test                               | [`requirements/tests.txt`]           | Paquetes para ejecutar los tests. Si se usa [Nox], no será necesario instalar estos paquetes a mano. Además se deben instalar los programas [pandoc], [aspell] y el [diccionario de español para aspell][aspell-es] para poder ejecutar los tests de documentación. |
  | Comparativa de Optimizadores       | [`requirements/hp_optimization.txt`] | Paquetes usados en la comparativa de optimizadores de hiper-parámetros.                                                 |
  | Comparativa de Frameworks para MLP | [`requirements/mlp_frameworks.txt`]  | Paquetes usados en la comparativa de frameworks para redes neuronales.                                                  |

  Para instalar cualquiera de los grupos de paquetes hay que ejecutar:

  ```shell
  pip install -r <nombre archivo>
  ```

  pudiendo sustituirse `<nombre archivo>` cualquiera de los anteriores.
  Si se quisiera instalar los paquetes sin usar un entorno virtual (no
  recomendado) se puede usar el siguiente comando:

  ```shell
  python3.7 -m pip install --user -r <nombre archivo>
  ```

  Es el mismo comando pero precedido por `python3.7 -m` para evitar problemas
  si tenemos otras versiones de Python instaladas en el sistema.

## Utilización

> **Nota:** Si se ha optado por usar un entorno virtual, debe ser activado
> usando uno de los comandos mostrados en la [tabla de la sección de
> instalación](#table1.1) antes de ejecutar cualquiera de los
> siguientes comandos.

Usando [Nox] podemos ejecutar el siguiente comando para construir la
documentación:

```shell
nox -e build-pdf
```

Para ejecutar los test:

```shell
nox -k test
```

Para pasar los distintos linters al código:

```shell
nox -e lint
```

Para obtener otras opciones posibles con Nox:

```shell
nox -l
```

## Desarrollo

En este proyecto se usa un metodología de desarrollo basada en test.

Se utilizará también un fichero de cambios basado en [Keep a Changelog]. Esta
información se guardará en el archivo [`CHANGELOG.md`].

## Frameworks

- [Keras] - librería para la creación y ejecución de redes neuronales.

- [DEAP] - librería de construcción de algoritmos evolutivos. Se utilizará ésta
  para optimizar los parámetros de las redes neuronales.

En la documentación se podrá encontrar una comparativa detallada con otras
bibliotecas similares y el por qué de la elección de éstas.

## Utilidades

- General:

  - [Sultan] - librería en Python para ejecutar procesos de manera cómoda.

- Automatización:

  - [Nox] - herramienta de automatización para ejecutar procesos como la
    construcción de la documentación o el lanzamiento de tests.

- Tests:

  - [pytest] - librería de Python para la ejecución de test que se usarán en la
    integración continua.

  - [pandoc] y [aspell] - utilidades para convertir la documentación del
    trabajo a texto plano y comprobar si hay errores ortográficos.

- Documentación:

  - [Pweave] - módulo de Python que permite mostrar salida de código
    directamente en LaTeX.

  - [TexLive] - generador de archivos PDFs a partir de Latex.

## Licencia

El código de este repositorio está liberado bajo la licencia
[GPLv3]. Para más información vea el archivo [`LICENSE`].

<!-- Archivos -->
[`requirements/prod.txt`]: ./requirements/prod.txt
[`requirements.txt`]: ./requirements.txt
[`requirements/docs.txt`]: ./requirements/docs.txt
[`requirements/tests.txt`]: ./requirements/tests.txt
[`requirements/hp_optimization.txt`]: ./requirements/hp_optimization.txt
[`requirements/mlp_frameworks.txt`]: ./requirements/mlp_frameworks.txt
[`CHANGELOG.md`]: ./CHANGELOG.md
[`LICENSE`]: ./LICENSE

<!-- Misceláneo -->
[python-downloads-url]: https://www.python.org/downloads/
[pip]: https://pypi.org/project/pip/
[venv]: https://docs.python.org/3/library/venv.html
[python-venv-pip-guide-url]: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
[pandoc]: https://pandoc.org/
[aspell]: http://aspell.net/
[aspell-es]: https://ftp.gnu.org/gnu/aspell/dict/es/
[Keep a Changelog]: https://keepachangelog.com/en/1.0.0/
[GPLv3]: https://www.gnu.org/licenses/gpl-3.0.en.html

<!-- Frameworks y utilidades -->
[Keras]: https://keras.io/
[DEAP]: https://deap.readthedocs.io/en/master/
[Sultan]: https://sultan.readthedocs.io/en/latest/
[Nox]: https://nox.thea.codes/en/stable/
[pytest]: https://docs.pytest.org/en/latest/
[aspell]: http://aspell.net/man-html/Introduction.html#Introduction
[PyPDF]: http://mstamy2.github.io/PyPDF2/
[Pweave]: http://mpastell.com/pweave/
[TexLive]: https://tug.org/texlive/

<!-- Insignias -->
[travis-badge]: https://travis-ci.org/lulivi/deep-g-prop.svg?branch=master
[travis-url]: https://travis-ci.org/lulivi/deep-g-prop
[license-badge]: https://img.shields.io/github/license/lulivi/deep-g-prop
[tag-badge]: https://img.shields.io/github/v/tag/lulivi/deep-g-prop
