# encoder_det

Analisis exploratorio de senales de un encoder rotatorio binario muestreadas a
500 Hz. El proyecto convierte datos MATLAB a Parquet, calcula la velocidad
neta del encoder y prepara filtros para estudiar cambios de comportamiento.

## Requisitos

- Python 3.12 o posterior
- Dependencias listadas en `requirements.txt`

## Instalacion

Desde la raiz del repositorio:

```bash
python -m venv .venv
```

Activa el entorno virtual y luego instala las dependencias:

```bash
python -m pip install -r requirements.txt
```

En Windows PowerShell, la activacion se realiza con:

```powershell
.venv\Scripts\Activate.ps1
```

## Flujo de trabajo

### 1. Convertir el dataset MATLAB

El script valida la matriz `resultado` del archivo MATLAB y genera
`dataset/rotary_encoder_500Hz.parquet`.

```bash
python eda/data_prep.py --fs 500
```

La salida contiene las columnas `ensayo`, `numero_muestra`, `avance` y
`retroceso`.

### 2. Calcular y filtrar la velocidad

La velocidad neta se define como `avance - retroceso`. Por defecto se aplica
una media movil de 100 ms por ensayo:

```bash
python eda/velocity_changes.py --method moving-average --window-ms 100
```

Tambien se puede aplicar un filtro Butterworth pasa-altos:

```bash
python eda/velocity_changes.py --method high-pass --cutoff-hz 1 --filter-order 4
```

Los resultados se guardan como archivos Parquet dentro de `dataset/`.

### 3. Explorar y visualizar

```bash
python eda/read_data.py
```

El script imprime estadisticas descriptivas y genera figuras en `eda/figures/`.

## Estructura

```text
encoder_det/
|-- dataset/                  # Datos MATLAB, Parquet y senales procesadas
|-- eda/
|   |-- data_prep.py          # Conversion y validacion del dataset
|   |-- read_data.py          # Exploracion y graficos
|   |-- velocity_changes.py   # Velocidad y filtrado por ensayo
|   `-- figures/              # Figuras generadas
|-- ideas.md                  # Notas metodologicas
|-- requirements.txt          # Dependencias de Python
`-- README.md
```

## Datos

El dataset de origen contiene senales binarias de avance y retroceso,
organizadas por ensayo. La frecuencia de muestreo predeterminada es 500 Hz.

## Estado del proyecto

El repositorio contiene el pipeline inicial de preparacion, exploracion y
filtrado. La deteccion automatica de cambios de comportamiento es el siguiente
paso de desarrollo descrito en `ideas.md`.
