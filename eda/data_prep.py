"""Convierte el dataset MATLAB del encoder a formato Parquet."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "dataset"
MAT_PATH = DATASET_DIR / "rotary_encoder_500Hz.mat"
PARQUET_PATH = MAT_PATH.with_suffix(".parquet")
VARIABLE_NAME = "resultado"
DEFAULT_SAMPLING_FREQUENCY_HZ = 500.0
COLUMN_NAMES = ("ensayo", "numero_muestra", "avance", "retroceso")


def prepare_dataframe(
    values: np.ndarray, sampling_frequency_hz: float
) -> pd.DataFrame:
    """Valida la matriz del encoder y asigna nombres y tipos adecuados."""
    if sampling_frequency_hz <= 0:
        raise ValueError("La frecuencia de muestreo debe ser mayor que cero.")
    if values.ndim != 2 or values.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f"Se esperaba una matriz de N x {len(COLUMN_NAMES)}, "
            f"pero se obtuvo {values.shape}."
        )
    if not np.isfinite(values).all():
        raise ValueError("El dataset contiene valores no finitos.")

    ensayo, tiempo_s, avance, retroceso = values.T
    if not np.equal(ensayo, np.floor(ensayo)).all() or (ensayo < 0).any():
        raise ValueError("La columna ensayo debe contener enteros no negativos.")
    if (ensayo > np.iinfo(np.uint32).max).any():
        raise ValueError("La columna ensayo excede el rango de uint32.")
    if not np.isin(avance, (0, 1)).all() or not np.isin(retroceso, (0, 1)).all():
        raise ValueError("Las columnas avance y retroceso deben contener solo 0 o 1.")

    sample_values = tiempo_s * sampling_frequency_hz
    sample_numbers = np.rint(sample_values)
    if not np.allclose(sample_values, sample_numbers):
        raise ValueError("El tiempo no corresponde a numeros enteros de muestra.")

    starts_trial = np.r_[True, ensayo[1:] != ensayo[:-1]]
    trial_starts = np.flatnonzero(starts_trial)
    trial_ids = np.cumsum(starts_trial, dtype=np.uint32) - 1
    sample_numbers = sample_numbers - sample_numbers[trial_starts][trial_ids]

    same_trial = ensayo[1:] == ensayo[:-1]
    if not np.equal(np.diff(sample_numbers)[same_trial], 1).all():
        raise ValueError(
            f"El tiempo no corresponde a una frecuencia de {sampling_frequency_hz:g} Hz."
        )

    return pd.DataFrame(
        {
            "ensayo": ensayo.astype("uint32"),
            "numero_muestra": sample_numbers.astype("uint32"),
            "avance": avance.astype("uint8"),
            "retroceso": retroceso.astype("uint8"),
        }
    )


def convert_mat_to_parquet(sampling_frequency_hz: float) -> Path:
    """Convierte la matriz configurada y devuelve la ruta del archivo Parquet."""
    mat_data = loadmat(MAT_PATH)
    if VARIABLE_NAME not in mat_data:
        available = [name for name in mat_data if not name.startswith("__")]
        raise KeyError(
            f"No se encontro '{VARIABLE_NAME}'. Variables disponibles: {available}"
        )

    dataframe = prepare_dataframe(mat_data[VARIABLE_NAME], sampling_frequency_hz)
    dataframe.to_parquet(PARQUET_PATH, engine="pyarrow",
                         compression="zstd", index=False)

    print(f"Origen: {MAT_PATH}")
    print(f"Variable: {VARIABLE_NAME}")
    print(
        f"Dimensiones: {dataframe.shape[0]:,} filas x {dataframe.shape[1]} columnas")
    print(f"Frecuencia de muestreo (fs): {sampling_frequency_hz:g} Hz")
    print(f"Periodo de muestreo (Ts): {1 / sampling_frequency_hz:g} s")
    print("Columnas y tipos:")
    print(dataframe.dtypes.to_string())
    print(f"Archivo convertido: {PARQUET_PATH}")
    print(f"Tamano: {PARQUET_PATH.stat().st_size / (1024**2):.2f} MiB")
    return PARQUET_PATH


def main() -> None:
    """Lee la configuracion de muestreo y ejecuta la conversion."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python eda/data_prep.py\n"
            "  python eda/data_prep.py --fs 500"
        ),
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=DEFAULT_SAMPLING_FREQUENCY_HZ,
        help=f"Frecuencia de muestreo en Hz (default: {DEFAULT_SAMPLING_FREQUENCY_HZ:g}).",
    )
    arguments = parser.parse_args()
    convert_mat_to_parquet(arguments.fs)


if __name__ == "__main__":
    main()
