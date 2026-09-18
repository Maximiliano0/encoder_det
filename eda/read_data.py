"""Explora el dataset convertido del encoder y genera graficos descriptivos."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "dataset" / "rotary_encoder_500Hz.parquet"
VELOCITY_DATA_PATH = PROJECT_DIR / "dataset" / "rotary_encoder_velocity.parquet"
HIGH_PASS_DATA_PATH = PROJECT_DIR / "dataset" / "rotary_encoder_velocity_highpass.parquet"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
DEFAULT_SAMPLING_FREQUENCY_HZ = 500.0
REQUIRED_COLUMNS = {"ensayo", "numero_muestra", "avance", "retroceso"}
MOVING_AVERAGE_REQUIRED_COLUMNS = {
    "ensayo",
    "numero_muestra",
    "velocidad_neta",
    "velocidad_suavizada",
}
HIGH_PASS_REQUIRED_COLUMNS = {
    "ensayo",
    "numero_muestra",
    "velocidad_neta",
    "velocidad_pasa_altos",
}


def print_exploration(
    dataframe: pd.DataFrame, sampling_frequency_hz: float
) -> None:
    """Imprime estructura, calidad y estadisticas descriptivas del dataset."""
    print("\n=== EXPLORACION DEL DATASET ===")
    print(f"Archivo: {DATA_PATH}")
    print(f"Frecuencia de muestreo (fs): {sampling_frequency_hz:g} Hz")
    print(f"Periodo de muestreo (Ts): {1 / sampling_frequency_hz:g} s")
    print(f"Dimensiones: {dataframe.shape[0]:,} filas x {dataframe.shape[1]} columnas")
    print(f"Memoria en pandas: {dataframe.memory_usage(deep=True).sum() / (1024**2):.2f} MiB")

    print("\nTipos de datos:")
    print(dataframe.dtypes.to_string())
    print("\nPrimeras 5 filas:")
    print(dataframe.head().to_string(index=False))
    print("\nUltimas 5 filas:")
    print(dataframe.tail().to_string(index=False))

    print("\nValores faltantes por columna:")
    print(dataframe.isna().sum().to_string())
    print("\nValores unicos por columna:")
    print(dataframe.nunique(dropna=False).to_string())
    print("\n=== ESTADISTICA DESCRIPTIVA ===")
    print(dataframe.describe().T.to_string())
    print("\nMatriz de correlacion:")
    print(dataframe.corr(numeric_only=True).round(4).to_string())


def create_plots(
    dataframe: pd.DataFrame, show: bool, sampling_frequency_hz: float
) -> None:
    """Genera y guarda graficos de senales, distribuciones y correlacion."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    numeric_data = dataframe.select_dtypes(include="number")
    if numeric_data.empty:
        print("No hay columnas numericas para graficar.")
        return

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes = np.atleast_1d(axes)
    signal_columns = ("avance", "retroceso")
    for _, trial_data in dataframe.groupby("ensayo", sort=False):
        for axis, column in zip(axes, signal_columns):
            axis.plot(
                trial_data["numero_muestra"],
                trial_data[column],
                linewidth=0.4,
                alpha=0.35,
            )
    for axis, column in zip(axes, signal_columns):
        axis.set_ylabel(column)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Numero de muestra (n = tiempo_s * fs)")
    figure.suptitle(
        f"Senales completas del encoder por ensayo (fs = {sampling_frequency_hz:g} Hz)"
    )
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "signals.png", dpi=150)

    numeric_data.hist(figsize=(11, 7), bins=40, edgecolor="white")
    plt.suptitle("Distribuciones por columna")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "distributions.png", dpi=150)

    correlation = numeric_data.corr()
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(correlation.columns)), correlation.columns)
    for row in range(len(correlation.index)):
        for column in range(len(correlation.columns)):
            axis.text(column, row, f"{correlation.iloc[row, column]:.2f}", ha="center", va="center")
    axis.set_title("Matriz de correlacion")
    figure.colorbar(image, ax=axis, label="Correlacion")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "correlation.png", dpi=150)

    print(f"\nGraficos guardados en: {FIGURES_DIR}")
    if show:
        plt.show()
    else:
        plt.close("all")


def create_trial_plot(
    dataframe: pd.DataFrame,
    trial_number: int,
    show: bool,
    sampling_frequency_hz: float,
) -> None:
    """Grafica señales y espectros mejorados para un ensayo.

    La señal temporal se grafica directamente, sin ventana ni zero-padding.
    El espectro se calcula después de quitar la media, aplicar una ventana
    Hann y usar zero-padding para densificar su visualización. La media se
    conserva en el espectro y se muestra la magnitud normalizada en dB
    relativos a su máximo en toda la banda Nyquist.
    """
    trial_data = dataframe[dataframe["ensayo"] == trial_number].copy()
    if trial_data.empty:
        available_trials = dataframe["ensayo"].unique()
        raise ValueError(
            f"No existe el ensayo {trial_number}. "
            f"Ensayos disponibles: {available_trials.min()} a {available_trials.max()}."
        )

    sample_numbers = trial_data["numero_muestra"].to_numpy()
    signal_columns = ("avance", "retroceso")
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), sharex="col")

    for row, column in enumerate(signal_columns):
        signal = trial_data[column].to_numpy(dtype=float)
        dc_value = signal.mean()
        signal_without_dc = signal - dc_value
        rms_without_dc = np.sqrt(np.mean(signal_without_dc**2))
        rms_dc = np.sqrt(np.mean(np.full(signal.size, dc_value) ** 2))
        rms_complete = np.sqrt(np.mean(signal**2))

        window = np.hanning(signal.size)
        nfft = 2 ** int(np.ceil(np.log2(signal.size)))
        frequency = np.fft.rfftfreq(nfft, d=1 / sampling_frequency_hz)
        spectrum = np.abs(
            np.fft.rfft(signal * window, n=nfft)
        )
        spectrum_db = 20 * np.log10(
            np.maximum(spectrum / spectrum.max(), 1e-12)
        )

        axes[row, 0].step(
            sample_numbers,
            signal,
            where="post",
            linewidth=0.9,
        )
        axes[row, 0].set_ylabel(f"{column.capitalize()} (0/1)")
        axes[row, 0].set_ylim(-0.1, 1.1)
        axes[row, 0].set_title(
            f"{column.capitalize()} original vs. número de muestra (sin padding)"
        )
        axes[row, 0].grid(alpha=0.3, linestyle="--")

        axes[row, 1].plot(
            frequency,
            spectrum_db,
            linewidth=0.8,
            color="tab:orange",
            label=(
                f"RMS señal sin DC: {rms_without_dc:.3f}\n"
                f"RMS componente DC: {rms_dc:.3f}\n"
                f"RMS señal completa: {rms_complete:.3f}"
            ),
        )
        axes[row, 1].set_ylabel("Magnitud normalizada (dB)")
        axes[row, 1].set_title(
            f"FFT de {column}: Hann, con media, zero-padding, dB relativos al máximo"
        )
        axes[row, 1].set_ylim(spectrum_db.min(), 0)
        axes[row, 1].grid(alpha=0.3, linestyle="--")
        axes[row, 1].legend(loc="upper right", framealpha=0.9)

        print(
            f"Ensayo {trial_number}, {column}: "
            f"RMS señal sin DC={rms_without_dc:.6f}, "
            f"RMS componente DC={rms_dc:.6f}, "
            f"RMS señal completa={rms_complete:.6f}"
        )

    axes[1, 0].set_xlabel("Número de muestra")
    axes[1, 1].set_xlabel("Frecuencia (Hz), banda completa hasta Nyquist")
    figure.suptitle(
        f"Ensayo {trial_number}: señales y módulo de FFT "
        f"(fs = {sampling_frequency_hz:g} Hz)"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / f"ensayo_{trial_number}_signals_fft.png"
    figure.savefig(output_path, dpi=150)
    print(f"Figura guardada en: {output_path}")
    if show:
        plt.show()
    else:
        plt.close(figure)


def create_velocity_trial_plot(
    dataframe: pd.DataFrame,
    trial_number: int,
    show: bool,
    sampling_frequency_hz: float,
    filtered_column: str = "velocidad_suavizada",
    output_suffix: str = "velocity",
    filtered_label: str = "velocidad suavizada",
) -> None:
    """Grafica velocidad neta y filtrada, junto con sus espectros."""
    trial_data = dataframe[dataframe["ensayo"] == trial_number].copy()
    if trial_data.empty:
        available_trials = dataframe["ensayo"].unique()
        raise ValueError(
            f"No existe el ensayo {trial_number}. "
            f"Ensayos disponibles: {available_trials.min()} a {available_trials.max()}."
        )

    sample_numbers = trial_data["numero_muestra"].to_numpy()
    signal_columns = ("velocidad_neta", filtered_column)
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), sharex="col")

    for row, column in enumerate(signal_columns):
        signal = trial_data[column].to_numpy(dtype=float)
        dc_value = signal.mean()
        signal_without_dc = signal - dc_value
        rms_without_dc = np.sqrt(np.mean(signal_without_dc**2))
        rms_dc = np.sqrt(np.mean(np.full(signal.size, dc_value) ** 2))
        rms_complete = np.sqrt(np.mean(signal**2))

        window = np.hanning(signal.size)
        nfft = 2 ** int(np.ceil(np.log2(signal.size)))
        frequency = np.fft.rfftfreq(nfft, d=1 / sampling_frequency_hz)
        spectrum = np.abs(np.fft.rfft(signal * window, n=nfft))
        spectrum_db = 20 * np.log10(np.maximum(spectrum / spectrum.max(), 1e-12))

        axes[row, 0].plot(sample_numbers, signal, linewidth=0.9)
        axes[row, 0].set_ylabel(
            f"{column.replace('_', ' ').capitalize()}\n(pulsos/muestra)"
        )
        axes[row, 0].set_title(
            f"{column.replace('_', ' ').capitalize()} vs. número de muestra"
        )
        axes[row, 0].grid(alpha=0.3, linestyle="--")

        axes[row, 1].plot(
            frequency,
            spectrum_db,
            linewidth=0.8,
            color="tab:orange",
            label=(
                f"RMS señal sin DC: {rms_without_dc:.3f}\n"
                f"RMS componente DC: {rms_dc:.3f}\n"
                f"RMS señal completa: {rms_complete:.3f}"
            ),
        )
        axes[row, 1].set_ylabel("Magnitud normalizada (dB)")
        axes[row, 1].set_title(f"FFT de {column.replace('_', ' ')}")
        axes[row, 1].set_ylim(spectrum_db.min(), 0)
        axes[row, 1].grid(alpha=0.3, linestyle="--")
        axes[row, 1].legend(loc="upper right", framealpha=0.9)

        print(
            f"Ensayo {trial_number}, {column}: "
            f"RMS señal sin DC={rms_without_dc:.6f}, "
            f"RMS componente DC={rms_dc:.6f}, "
            f"RMS señal completa={rms_complete:.6f}"
        )

    axes[1, 0].set_xlabel("Número de muestra")
    axes[1, 1].set_xlabel("Frecuencia (Hz), banda completa hasta Nyquist")
    figure.suptitle(
        f"Ensayo {trial_number}: velocidad neta y {filtered_label}, módulo de FFT "
        f"(fs = {sampling_frequency_hz:g} Hz)"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / f"ensayo_{trial_number}_{output_suffix}_fft.png"
    figure.savefig(output_path, dpi=150)
    print(f"Figura guardada en: {output_path}")
    if show:
        plt.show()
    else:
        plt.close(figure)


def main() -> None:
    """Carga el dataset, selecciona un ensayo y crea sus figuras."""
    parser = argparse.ArgumentParser(
        description="Exploracion del dataset del encoder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python eda/read_data.py 1\n"
            "  python eda/read_data.py 1 --input dataset/otro.parquet\n"
            "  python eda/read_data.py 1 --velocity-mav\n"
            "  python eda/read_data.py 1 --velocity-hpf --show"
        ),
    )
    parser.add_argument(
        "ensayo",
        nargs="?",
        type=int,
        help="Numero de ensayo que se desea visualizar.",
    )
    parser.add_argument("--show", action="store_true", help="Muestra las figuras en pantalla.")
    parser.add_argument(
        "--input",
        type=Path,
        help="Archivo Parquet de entrada. Si se omite, se usa el dataset del modo elegido.",
    )
    velocity_group = parser.add_mutually_exclusive_group()
    velocity_group.add_argument(
        "--velocity-mav",
        action="store_const",
        const="moving-average",
        dest="velocity_method",
        help="Visualiza velocidad neta y suavizada.",
    )
    velocity_group.add_argument(
        "--velocity-hpf",
        action="store_const",
        const="high-pass",
        dest="velocity_method",
        help="Visualiza velocidad neta y filtrada con pasa altos.",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=DEFAULT_SAMPLING_FREQUENCY_HZ,
        help=f"Frecuencia de muestreo en Hz (default: {DEFAULT_SAMPLING_FREQUENCY_HZ:g}).",
    )
    arguments = parser.parse_args()
    if arguments.fs <= 0:
        parser.error("--fs debe ser mayor que cero.")

    velocity_method = arguments.velocity_method
    if velocity_method == "moving-average":
        data_path = VELOCITY_DATA_PATH
        required_columns = MOVING_AVERAGE_REQUIRED_COLUMNS
    elif velocity_method == "high-pass":
        data_path = HIGH_PASS_DATA_PATH
        required_columns = HIGH_PASS_REQUIRED_COLUMNS
    else:
        data_path = DATA_PATH
        required_columns = REQUIRED_COLUMNS
    if arguments.input is not None:
        data_path = arguments.input
    if not data_path.exists():
        raise FileNotFoundError(
            f"No existe {data_path}. "
            "Ejecuta primero el script de preparación correspondiente."
        )

    dataframe = pd.read_parquet(data_path)
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"Faltan columnas requeridas: {sorted(missing_columns)}. "
            "Comprueba que el dataset fue preparado correctamente."
        )
    if arguments.ensayo is None:
        try:
            trial_number = int(input("Ingrese el numero de ensayo que desea visualizar: "))
        except ValueError as error:
            raise ValueError("El numero de ensayo debe ser un entero.") from error
    else:
        trial_number = arguments.ensayo

    if velocity_method == "moving-average":
        create_velocity_trial_plot(
            dataframe, trial_number, arguments.show, arguments.fs
        )
    elif velocity_method == "high-pass":
        create_velocity_trial_plot(
            dataframe,
            trial_number,
            arguments.show,
            arguments.fs,
            filtered_column="velocidad_pasa_altos",
            output_suffix="velocity_highpass",
            filtered_label="velocidad pasa altos",
        )
    else:
        create_trial_plot(dataframe, trial_number, arguments.show, arguments.fs)


if __name__ == "__main__":
    main()
