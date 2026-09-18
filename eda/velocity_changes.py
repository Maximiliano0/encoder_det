"""Calcula y suaviza la velocidad neta de las señales del encoder."""

import argparse
from pathlib import Path

import pandas as pd
from scipy.signal import butter, sosfiltfilt


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_DIR / "dataset" / "rotary_encoder_500Hz.parquet"
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "dataset" / "rotary_encoder_velocity.parquet"
DEFAULT_SAMPLING_FREQUENCY_HZ = 500.0
DEFAULT_WINDOW_MS = 100.0
DEFAULT_CUTOFF_HZ = 1.0
DEFAULT_FILTER_ORDER = 4
REQUIRED_COLUMNS = {"ensayo", "avance", "retroceso"}


def calculate_velocity(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Añade la velocidad neta en pulsos por muestra."""
    result = dataframe.copy()
    result["velocidad_neta"] = (
        result["avance"].astype(float) - result["retroceso"].astype(float)
    )
    return result


def smooth_velocity(
    dataframe: pd.DataFrame, window_samples: int
) -> pd.DataFrame:
    """Añade una media móvil centrada, reiniciada en cada ensayo."""
    if window_samples < 1:
        raise ValueError("La ventana debe contener al menos una muestra.")

    result = dataframe.copy()
    result["velocidad_suavizada"] = result.groupby("ensayo", sort=False)[
        "velocidad_neta"
    ].transform(
        lambda values: values.rolling(
            window=window_samples, center=True, min_periods=1
        ).mean()
    )
    return result


def highpass_velocity(
    dataframe: pd.DataFrame,
    sampling_frequency_hz: float,
    cutoff_hz: float,
    filter_order: int,
) -> pd.DataFrame:
    """Añade la velocidad filtrada con un Butterworth pasa altos por ensayo."""
    if sampling_frequency_hz <= 0:
        raise ValueError("La frecuencia de muestreo debe ser mayor que cero.")
    if not 0 < cutoff_hz < sampling_frequency_hz / 2:
        raise ValueError(
            "La frecuencia de corte debe estar entre cero y la frecuencia de Nyquist."
        )
    if filter_order < 1:
        raise ValueError("El orden del filtro debe ser al menos uno.")

    second_order_sections = butter(
        filter_order,
        cutoff_hz,
        btype="highpass",
        fs=sampling_frequency_hz,
        output="sos",
    )
    result = dataframe.copy()

    def apply_filter(values: pd.Series) -> pd.Series:
        try:
            filtered_values = sosfiltfilt(second_order_sections, values.to_numpy())
        except ValueError as error:
            raise ValueError(
                "Cada ensayo debe tener suficientes muestras para aplicar el filtro "
                "pasa altos."
            ) from error
        return pd.Series(filtered_values, index=values.index)

    result["velocidad_pasa_altos"] = result.groupby("ensayo", sort=False)[
        "velocidad_neta"
    ].transform(apply_filter)
    return result


def process_dataset(
    input_path: Path,
    output_path: Path,
    sampling_frequency_hz: float,
    window_samples: int,
    method: str = "moving-average",
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    filter_order: int = DEFAULT_FILTER_ORDER,
) -> pd.DataFrame:
    """Carga, procesa y guarda el dataset del encoder."""
    if sampling_frequency_hz <= 0:
        raise ValueError("La frecuencia de muestreo debe ser mayor que cero.")

    dataframe = pd.read_parquet(input_path)
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing_columns)}.")

    processed = calculate_velocity(dataframe)
    if method == "moving-average":
        processed = smooth_velocity(processed, window_samples)
    elif method == "high-pass":
        processed = highpass_velocity(
            processed, sampling_frequency_hz, cutoff_hz, filter_order
        )
    else:
        raise ValueError(f"Metodo de filtrado no soportado: {method}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(output_path, engine="pyarrow", compression="zstd", index=False)
    return processed


def main() -> None:
    """Ejecuta el calculo de velocidad neta y el filtrado seleccionado."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python eda/velocity_changes.py --method moving-average --window-ms 100\n"
            "  python eda/velocity_changes.py --method high-pass --cutoff-hz 1 "
            "--filter-order 4"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Archivo Parquet de entrada (default: {DEFAULT_INPUT_PATH}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Archivo Parquet de salida (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=DEFAULT_SAMPLING_FREQUENCY_HZ,
        help=f"Frecuencia de muestreo en Hz (default: {DEFAULT_SAMPLING_FREQUENCY_HZ:g}).",
    )
    parser.add_argument(
        "--method",
        choices=("moving-average", "high-pass"),
        default="moving-average",
        help="Filtro a aplicar (default: moving-average).",
    )
    window_group = parser.add_mutually_exclusive_group()
    window_group.add_argument(
        "--window-ms",
        type=float,
        default=DEFAULT_WINDOW_MS,
        help=f"Duracion de la ventana en ms (default: {DEFAULT_WINDOW_MS:g}).",
    )
    window_group.add_argument(
        "--window-samples",
        type=int,
        help="Tamano de la ventana directamente en muestras.",
    )
    parser.add_argument(
        "--cutoff-hz",
        type=float,
        default=DEFAULT_CUTOFF_HZ,
        help=f"Frecuencia de corte del pasa altos en Hz (default: {DEFAULT_CUTOFF_HZ:g}).",
    )
    parser.add_argument(
        "--filter-order",
        type=int,
        default=DEFAULT_FILTER_ORDER,
        help=f"Orden del filtro pasa altos (default: {DEFAULT_FILTER_ORDER}).",
    )
    arguments = parser.parse_args()

    if arguments.fs <= 0:
        parser.error("--fs debe ser mayor que cero.")
    if arguments.cutoff_hz <= 0 or arguments.cutoff_hz >= arguments.fs / 2:
        parser.error("--cutoff-hz debe estar entre cero y fs/2.")
    if arguments.filter_order < 1:
        parser.error("--filter-order debe ser al menos uno.")
    if arguments.window_samples is not None:
        window_samples = arguments.window_samples
    else:
        if arguments.window_ms <= 0:
            parser.error("--window-ms debe ser mayor que cero.")
        window_samples = max(1, round(arguments.window_ms * arguments.fs / 1000))
    if window_samples < 1:
        parser.error("La ventana debe contener al menos una muestra.")
    if not arguments.input.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {arguments.input}")

    processed = process_dataset(
        arguments.input,
        arguments.output,
        arguments.fs,
        window_samples,
        arguments.method,
        arguments.cutoff_hz,
        arguments.filter_order,
    )
    print(f"Filas procesadas: {len(processed):,}")
    print(f"Frecuencia de muestreo: {arguments.fs:g} Hz")
    print(f"Metodo: {arguments.method}")
    if arguments.method == "moving-average":
        print(
            f"Ventana: {window_samples} muestras "
            f"({window_samples / arguments.fs * 1000:g} ms)"
        )
    else:
        print(
            f"Frecuencia de corte: {arguments.cutoff_hz:g} Hz; "
            f"orden: {arguments.filter_order}"
        )
    print(f"Archivo generado: {arguments.output}")


if __name__ == "__main__":
    main()
