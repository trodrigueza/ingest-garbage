# Ingesta de Planillas de Basuras 

Pipeline asíncrono en Python para digitalizar planillas manuscritas usando **google-genai**, **Pydantic** y **LangGraph**. Extrae los datos a un esquema tipado, ejecuta consenso multi-modelo, marca discrepancias con `*` y exporta a Excel.

## Requisitos
- Python 3.11+ (3.12 recomendado).
- Dependencias: `pip install google-genai langgraph pydantic pandas pymupdf`.
- Variables de entorno:
  - `GOOGLE_API_KEY` (o `GENAI_API_KEY`): clave de Gemini.
  - Opcionales:
    - `GENAI_CONSENSUS_MODELS` para definir el orden de modelos en consenso (coma-separado).
    - `GENAI_MODEL_ATTEMPTS` número de intentos por modelo (default: 1).
    - `GENAI_RETRY_ON_ANOMALY` reintento por anomalías en sanity check (default: false).
    - `GENAI_PDF_ZOOM` escala de render para PDFs (default: 1.3).
    - `GENAI_BATCH_CONCURRENCY` concurrencia máxima en batch (default: 3).

## Modelos
- Por defecto, una sola pasada con `models/gemini-3-pro-preview` para acelerar.
- Puedes sobreescribir:
  - `--model-pro` (thinking) p.ej. `models/gemini-3-pro-preview`.
  - `--model-flash` (rápido) p.ej. `models/gemini-flash-latest`.
  - `GENAI_CONSENSUS_MODELS` si quieres consenso multi-modelo (no se deduplica). Ejemplo:
    ```bash
    export GENAI_CONSENSUS_MODELS="models/gemini-3-pro-preview,models/gemini-2.5-pro,models/gemini-flash-latest"
    ```

## Prompt de extracción (en `main.py`)
El prompt integrado indica:
- Detectar Mes/Año de cabecera.
- Encabezados lógicos: Semana, Día, Fecha, Kg Orgánicos (bolsa verde), Kg Reciclaje (bolsa blanca), Kg No aprovechables, Notas.
- Números: solo el valor numérico (ignorar “kilo/kilos/k”).
- Celdas vacías o con guiones: `null` en el JSON.
- Texto especial (ej. “Festivo”): ubicar en la primera columna de kilos disponible o en notas.
- No inventar; respetar tachaduras/correcciones. Segunda pasada opcional con “mejor contraste” si hay anomalías.

## Flujo (LangGraph)
1. **Ingesta**: lee la imagen o renderiza páginas de PDF.
2. **VisionExtraction**: llama a Gemini según `consensus_models` (secuencial, reintento simple). Se omiten modelos sin soporte de `response_schema`.
3. **Consensus**: compara salidas, arma un `PlanillaDigitalizada` final y registra discrepancias (por campo/fila). Si los valores difieren, ese campo se marca con `*` en la exportación.
4. **SanityCheck**: verifica confianza < 0.8, duplicados de fecha (ignorando vacías) y outliers numéricos; si falla y `GENAI_RETRY_ON_ANOMALY` está activo, reintenta con mejora de contraste; si no, exporta igualmente.
5. **Export**: genera `outputs/planilla_<mes>_<anio>.xlsx` con hojas:
   - `registros`: datos; campos con discrepancias llevan `*`.
   - `resumen`: metadatos y notas de auditoría.
   - `inconsistencias`: detalle de diferencias por campo/fila.

## Uso
```bash
export GOOGLE_API_KEY="tu_api_key"
python3.11 main.py <ruta_imagen_o_pdf> \
  --model-pro models/gemini-3-pro-preview \
  --model-flash models/gemini-flash-latest
```
Salida esperada:
- Excel en `outputs/planilla_<mes>_<anio>.xlsx`.
- Si no hay discrepancias entre modelos, no se añaden `*`. Los `*` indican campos con valores distintos entre extracciones.

### Batch (carpeta con múltiples archivos)
```bash
export GOOGLE_API_KEY="tu_api_key"
python3.11 main.py --dir ./carpeta_imagenes \
  --model-pro models/gemini-3-pro-preview \
  --model-flash models/gemini-flash-latest
# También puedes pasar el directorio como argumento principal:
# python3.11 main.py ./carpeta_imagenes --model-pro ... --model-flash ...
```
Genera `outputs/planillas_consolidadas.xlsx` con:
- `registros`: todas las filas de todas las imágenes/PDFs (columna `archivo` incluida); celdas con discrepancias llevan `*`.
- `resumen`: metadatos por archivo (confianza, observaciones, notas de auditoría).
- `inconsistencias`: detalle de diferencias por campo/fila/modelo.
Nota: los PDFs se renderizan por página y se listan como `archivo.pdf#page-N`.

## Web UI mínima (Streamlit)
- Requisitos extra: `pip install streamlit`.
- Ejecuta: `streamlit run app.py`
- En la UI ingresa la API Key, selecciona modelos (pro y flash), sube imágenes (jpg/png) o PDFs y descarga el Excel consolidado.

## Notas
- Temperatura fija en 0.0 para reproducibilidad.
- Si usas modelos sin soporte de JSON/schema (ej. `...flash-image`), recibirás 400; evita incluirlos en `GENAI_CONSENSUS_MODELS`.
- Las fechas se admiten opcionales para no bloquear; si el modelo deja la fecha vacía, no se marca duplicado ni se invalida la extracción.
