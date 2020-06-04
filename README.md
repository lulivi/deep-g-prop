# Trabajo de Fin de Grado: *DeepGProp*

> Optimización de Redes Neuronales con Algoritmos Genéticos

[![travis-badge]][travis-url]
[![license-badge]][LICENSE]
[![tag-badge]][CHANGELOG.md]

**Autor(a): Luis Liñán Villafranca**

**Tutor(a)(es): Juan Julián Merelo Guervós**

**Índice**
- [Instalación](#instalaci%c3%b3n)
- [Utilización](#utilizaci%c3%b3n)
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
  usar [pip]. He dividido los paquetes en dos grupos:

  - Paquetes de desarrollo: localizados en el archivo [requirements.txt] son
    los necesarios para crear la documentación y ejecutar el código que se
    encuentra en el repositorio.

  - Paquetes de testing: localizados en el archivo [test_requirements.txt] son
    opcionales para el desarrollo pero necesarios para ejecutar los test.

    Además se deben instalar los programas [pandoc], [aspell] y el
    [diccionario de español para `aspell`][aspell-es]. Para esto puede usar el
    gestor de paquetes que desee.

  Para instalar cualquiera de los dos grupos de paquetes hay que ejecutar:

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

Usando [Invoke] podemos usar lo siguiente para construir la documentación:

```shell
inv docs.pdf
```

Para ejecutar los test:

```shell
inv tests.go
```

Para obtener otras opciones posibles con `invoke`:

```shell
inv -l
```

## Desarrollo

En este proyecto se usa un metodología de desarrollo basada en test.

Se utilizará también un fichero de cambios basado en [Keep a Changelog]. Esta
información se guardará en el archivo [CHANGELOG.md].

## Frameworks

- [Keras] - librería para la creación y ejecución de redes neuronales. Se ha
  elegido ésta frente a otras librerías por su sencillez de uso, buena
  documentación y soporte (dada la gran comunidad que la utiliza).

- [DEAP] - librería de construcción de algoritmos evolutivos. Se utilizará ésta
  para optimizar los parámetros de las redes neuronales.

En la documentación se podrá encontrar una comparativa detallada con otras
bibliotecas similares y el por qué de la elección de éstas.

## Utilidades

- General:

  - [Sultan] - librería en Python para ejecutar procesos de manera cómoda.

- Automatización:

  - [Invoke] - utilidad para ejecutar procesos como la construcción de la
    documentación.

- Tests:

  - [pytest] - librería de Python para la ejecución de test que se usarán en la
    integración continua.

  - [pandoc] y [aspell] - utilidades para convertir la documentación del
    trabajo a texto plano y comprobar si hay errores ortográficos.

  - [PyPDF] - otra utilidad para comprobar distintas características del PDF
    resultante de documentación.

- Documentación:

  - [Pweave] - módulo de Python que permite mostrar salida de código
    directamente en LaTeX.

  - [TexLive] - generador de archivos PDFs a partir de Latex.

## Licencia

El código de este repositorio está liberado bajo la licencia
[GPLv3]. Para más información vea el archivo [LICENSE].

<!-- Archivos -->
[CHANGELOG.md]: ./CHANGELOG.md
[LICENSE]: ./LICENSE

<!-- Misceláneo -->
[python-downloads-url]: https://www.python.org/downloads/
[pip]: https://pypi.org/project/pip/
[venv]: https://docs.python.org/3/library/venv.html
[python-venv-pip-guide-url]: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
[requirements.txt]: ./requirements.txt
[test_requirements.txt]: ./test_requirements.txt
[pandoc]: https://pandoc.org/
[aspell]: http://aspell.net/
[aspell-es]: https://ftp.gnu.org/gnu/aspell/dict/es/
[Keep a Changelog]: https://keepachangelog.com/en/1.0.0/
[GPLv3]: https://www.gnu.org/licenses/gpl-3.0.en.html

<!-- Frameworks y utilidades -->
[Keras]: https://keras.io/
[DEAP]: https://deap.readthedocs.io/en/master/
[Sultan]: https://sultan.readthedocs.io/en/latest/
[Invoke]: http://docs.pyinvoke.org/en/1.2/
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
