# Ideas: detección de cambios de comportamiento en señales de encoder

Para señales de encoder binarias (`avance`/`retroceso` a 500 Hz) la clave es primero convertirlas en una **señal de velocidad**, y después aplicar detección de cambios sobre ella. Trabajar directamente con los pulsos 0/1 (STFT o wavelet sobre la señal cruda) mostrará principalmente armónicos de la tasa de pulsos, no los cambios de comportamiento.

## 1. Construir la señal de velocidad

- Velocidad neta con signo: $v[n] = \text{avance}[n] - \text{retroceso}[n]$ (pulsos por muestra), o por separado si interesa la dirección.
- Suavizar con ventana deslizante configurable (moving average): $\bar v[n] = \frac{1}{W}\sum_{k} v[n-k]$, en pulsos/s multiplicando por `fs/W`. Un kernel gaussiano evita los escalones del promedio rectangular.
- Posición acumulada $x[n] = \sum v$ para ver la trayectoria del ensayo; los cambios de pendiente son cambios de velocidad.

## 2. Detectar cambios abruptos (recomendado empezar aquí)

- **Diferencia de medias en dos ventanas adyacentes**: $d[n] = \bar v_{\text{post}}[n] - \bar v_{\text{pre}}[n]$; umbral con z-score robusto (MAD) por ensayo. Equivale a un wavelet Haar a una escala.
- **CUSUM** (Page): acumula desvíos respecto a la media local y dispara cuando supera un umbral; muy bueno para cambios de nivel bruscos y es online.
- **Changepoint detection** con penalización (PELT, Binary Segmentation) sobre $\bar v$: devuelve directamente los instantes de cambio de media/varianza. Librería `ruptures` (no está en el entorno; habría que instalarla).
- **Aceleración**: derivada de $\bar v$ (`np.gradient`) y detección de picos con `scipy.signal.find_peaks` con umbral de prominencia.

## 3. Multiescala / tiempo-frecuencia

- **Wavelet Haar (DWT o CWT)** sobre $\bar v$: los coeficientes grandes a varias escalas marcan cambios abruptos y dan la escala temporal del cambio. Es la extensión natural del método 2 sin elegir una única ventana.
- **CWT Morlet o STFT** sobre $\bar v$: útil si se esperan ritmos (p. ej. periodicidad de pasos o arranque/parada oscilatorios), menos para escalones. Con ventanas de 0.5–2 s se obtiene resolución razonable.

## 4. Segmentación por estados

- **HMM** de 2–3 estados (quieto / lento / rápido) sobre $\bar v$: los cambios de estado son los "momentos" de interés, y además da duraciones y probabilidades. Más robusto que umbrales fijos si hay variabilidad entre ensayos.

## 5. Validación

- Si se conoce el instante del estímulo por ensayo, alinear y promediar $\bar v$ (event-related average) y mirar la latencia de los cambios detectados respecto al estímulo.
- Comparar la distribución de instantes de cambio entre ensayos/condiciones.

## Pipeline sugerido

1. $v = \text{avance} - \text{retroceso}$ → suavizado gaussiano con `sigma` configurable (p. ej. 50–250 ms).
2. Detección: diferencia de medias pre/post + CUSUM, con umbral robusto por ensayo.
3. Refinar con Haar multiescala o PELT si el método simple da falsos positivos.
4. Gráfico por ensayo: posición acumulada, velocidad suavizada y marcadores verticales en los cambios detectados.

## Próximo paso

Implementar los pasos 1–2 y 4 como un script `eda/velocity_changes.py` con ventana y umbral configurables por argumento, siguiendo el mismo estilo de `read_data.py`.
