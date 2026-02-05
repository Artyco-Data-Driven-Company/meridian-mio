# 1. Sincronización con google-meridian

Este flujo permite mantener el repositorio `google-meridian-mio` alineado con el repositorio original de Google, incorporando sus actualizaciones de forma controlada.

- Añadir el repositorio original como *upstream* y obtener los últimos cambios del repositorio
    ```sh
    git remote add upstream https://github.com/google/meridian.git
    git fetch upstream main
    ```
- Cambiar a la rama `main` e integrar los cambios de Google
    ```sh
    git checkout main
    git merge upstream/main
    ```
    > *NOTA: En condiciones normales, no deberían producirse conflictos, ya que la rama main no incluye desarrollos propios.*

- Cambiar a la rama `meridian-mio-main` e integrar los cambios actualizados de `main` en `meridian-mio-main`
    ```sh
    git checkout meridian-mio-main
    git merge main
    ```
    > *NOTA: Si existen conflictos entre `main` y `meridian-mio-main`, deben resolverse manualmente antes de continuar.*

- Publicar los cambios en el repositorio remoto
    ```sh
    git push origin main
    git push origin meridian-mio-main
    ```


# 2. Uso de **google-meridian-mio** en proyectos reales

El paquete puede utilizarse directamente como librería en proyectos productivos, sin necesidad de clonar el repositorio.

### Instalación mediante Personal Access Token
```sh
GITHUB_TOKEN={github_token}  # Personal Access Token de GitHub
pip install git+https://$GITHUB_TOKEN@github.com/Artyco-Data-Driven-Company/meridian-mio.git@meridian-mio-main
```

### Instalación mediante SSH
```sh
pip install git+ssh://git@github.com/Artyco-Data-Driven-Company/meridian-mio.git@meridian-mio-main
```

Una vez instalado, se pueden tomar como referencia los ejemplos incluidos en el directorio `examples/`


# 3. **google-meridian-mio** para desarrollo y sandbox

Este modo está pensado para desarrollo activo, pruebas locales y contribuciones al código.

- Clonar el repositorio:
    ```sh
    git clone git@github.com:Artyco-Data-Driven-Company/meridian-mio.git
    cd meridian-mio
    ```
- Crear un entorno virtual y activarlo:
    ```sh
    python3 -m venv .venv_meridian
    source .venv_meridian/bin/activate
    ```
- Actualizar `pip` e instalar el paquete en modo editable con dependencias de desarrollo:
    ```sh
    pip install --upgrade pip
    pip install -e .'[dev]'
    ```
- Compilar estilos (necesario para el template `styles.css`):
    ```sh
    pip install libsass
    python setup.py build
    cp build/lib/meridian/analysis/templates/style.css meridian/analysis/templates/
    ```

Una vez instalado, se puede utilizar el notebook `sandbox/dev.ipynb` como punto de partida para pruebas locales y validaciones rápidas.
Alternativamente, es posible crear notebooks **(.ipynb)** o scripts de Python **(.py)** dentro de este entorno para experimentar con nuevas configuraciones o cambios en el código antes de integrarlos en proyectos reales.

### Instalación editable (-e)
La instalación en modo editable permite trabajar directamente sobre el código fuente del repositorio. En este modo, `pip` no copia los archivos a `site-packages`; en su lugar, crea un enlace simbólico que apunta directamente al directorio local del proyecto. (`./meridian`)
