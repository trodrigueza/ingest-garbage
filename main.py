import asyncio
import base64
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd
import fitz
from google import genai
from google.genai import types
from google.genai import errors
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError, field_validator


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger("planilla_ingesta")


# -------------------------
# Pydantic models (schema control)
# -------------------------

class FilaResiduo(BaseModel):
    semana: str = Field(..., description="Ej: 'Semana 1'")
    dia: str = Field(..., description="Día de la semana, normalizado (Lunes, Martes...)")
    fecha: Optional[str] = Field(None, description="Formato DD/MM/AAAA")
    kg_organicos: Optional[float] = Field(None, description="Valor numérico bolsa verde. Null si vacío.")
    kg_reciclaje: Optional[float] = Field(None, description="Valor numérico bolsa blanca. Null si vacío.")
    kg_no_aprovechables: Optional[float] = Field(None, description="Valor numérico. Null si vacío.")
    notas: Optional[str] = Field(None, description="Cualquier texto manuscrito adicional.")


class PlanillaDigitalizada(BaseModel):
    mes: str = Field(..., description="Mes detectado en el encabezado")
    anio: int = Field(..., description="Año del reporte")
    registros: List[FilaResiduo]
    observaciones_generales: Optional[str]
    confianza_extraccion: float = Field(..., description="Score de 0.0 a 1.0 de legibilidad")


# -------------------------
# LangGraph state
# -------------------------

class PipelineState(TypedDict, total=False):
    image_path: str
    image_bytes: bytes
    mime_type: str
    attempt: int
    enhance_contrast: bool
    planilla: PlanillaDigitalizada
    extractions: List[Dict[str, Any]]
    planilla_consensus: PlanillaDigitalizada
    discrepancies: List[Dict[str, Any]]
    needs_review: bool
    review_reason: str
    audit_notes: List[str]
    excel_path: str
    error: str


@dataclass
class PipelineConfig:
    output_dir: Path = Path("outputs")
    revision_dir: Path = Path("outputs/revision")
    # Defaults aligned to available models list
    model_pro: str = os.environ.get("GENAI_PRO_MODEL", "models/gemini-3-pro-preview")
    model_flash: str = os.environ.get("GENAI_FLASH_MODEL", "models/gemini-flash-latest")
    model_pro_fallbacks: List[str] = None  # type: ignore[assignment]
    consensus_models: List[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.model_pro_fallbacks is None:
            env_fallbacks = os.environ.get("GENAI_PRO_FALLBACKS", "")
            if env_fallbacks:
                self.model_pro_fallbacks = [m.strip() for m in env_fallbacks.split(",") if m.strip()]
            else:
                self.model_pro_fallbacks = [
                    "models/gemini-pro-latest",
                    "models/gemini-2.5-pro",
                    "models/gemini-2.5-flash",
                    "models/gemini-2.5-flash-lite",
                    "models/gemini-flash-latest",
                ]
        if self.consensus_models is None:
            env_models = os.environ.get("GENAI_CONSENSUS_MODELS", "")
            if env_models:
                self.consensus_models = [m.strip() for m in env_models.split(",") if m.strip()]
            else:
                # Secuencia: 1 pasada pro (thinking) y 1 pasada 2.5-pro para contraste
                self.consensus_models = [self.model_pro, "models/gemini-2.5-pro"]


# -------------------------
# Gemini helpers
# -------------------------

def build_client(api_key: Optional[str] = None) -> genai.Client:
    key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GENAI_API_KEY")
    if not key:
        raise RuntimeError("Falta la API key. Define GOOGLE_API_KEY o GENAI_API_KEY.")
    return genai.Client(api_key=key)


def _vision_prompt(enhance_contrast: bool) -> str:
    base_prompt = """
Actúa como un experto en digitalización y extracción de datos OCR especializado en formularios manuscritos de "Planillas de Registro Diario de Basuras".
Tu tarea: extraer la tabla al esquema JSON provisto (PlanillaDigitalizada) cumpliendo estas reglas:
- Identifica cabecera Mes/Año y usa esos valores en el JSON.
- Encabezados lógicos: Semana, Día, Fecha, Kg Orgánicos (bolsa verde), Kg Reciclaje (bolsa blanca), Kg No aprovechables, Notas.
- Números: escribe solo el valor numérico, ignora palabras como "kilo", "kilos", "k". Ej: "14 Kilos" -> 14.
- Celdas vacías o con guiones: deja el campo como null.
- Texto especial (ej: "Festivo"): ubícalo en la primera columna de kilos disponible para ese día y deja las otras vacías, o en notas si no encaja.
- No inventes valores; respeta tachaduras o correcciones manuscritas.
"""
    if enhance_contrast:
        base_prompt += (
            "Imagina que aplicaste un filtro de mejor contraste y nitidez. "
            "Relee números poco claros y prioriza fidelidad a lo escrito sobre la corrección."
        )
    return base_prompt.strip()


async def parse_planilla_with_gemini(
    client: genai.Client,
    model: str,
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    enhance_contrast: bool = False,
) -> PlanillaDigitalizada:
    prompt = _vision_prompt(enhance_contrast)
    response = await asyncio.to_thread(
        client.models.generate_content,
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    types.Part.from_text(text=prompt),
                ],
            )
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PlanillaDigitalizada,
            temperature=0.0,
        ),
    )
    if response.parsed is not None:
        return response.parsed  # Already validated to PlanillaDigitalizada

    # Fallback: try to parse raw candidate content manually to surface schema errors.
    try:
        candidates = getattr(response, "candidates", []) or []
        if candidates:
            parts = getattr(candidates[0], "content", None)
            if parts and getattr(parts, "parts", None):
                texts: List[str] = []
                for p in parts.parts:  # type: ignore[attr-defined]
                    if hasattr(p, "text") and p.text:
                        texts.append(p.text)
                if texts:
                    raw_text = "\n".join(texts)
                    try:
                        return PlanillaDigitalizada.model_validate_json(raw_text)
                    except Exception:
                        # If raw_text is not JSON, try loading as dict string
                        try:
                            data = json.loads(raw_text)
                            return PlanillaDigitalizada.model_validate(data)
                        except Exception as parse_exc:
                            LOGGER.warning("No se pudo parsear manualmente el candidato: %s", parse_exc)
    except Exception as exc:  # Defensive logging, but do not crash here
        LOGGER.warning("Fallo al inspeccionar candidato crudo: %s", exc)

    LOGGER.warning("Modelo %s devolvió resultado vacío o no conforme al esquema", model)
    return None  # Caller handles None


async def summarize_anomalies_with_flash(
    client: genai.Client, model: str, planilla: PlanillaDigitalizada, anomalies: List[str]
) -> str:
    response = await asyncio.to_thread(
        client.models.generate_content,
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text="Resume y justifica si estas anomalías son plausibles o errores:"),
                    types.Part.from_text(text=planilla.model_dump_json()),
                    types.Part.from_text(text=f"Anomalias: {anomalies}"),
                ],
            )
        ],
        config=types.GenerateContentConfig(temperature=0.2),
    )
    return response.text.strip()


# -------------------------
# Domain checks
# -------------------------

def detect_anomalies(planilla: PlanillaDigitalizada) -> List[str]:
    anomalies: List[str] = []
    if planilla.confianza_extraccion < 0.8:
        anomalies.append(f"Baja confianza: {planilla.confianza_extraccion:.2f}")

    fechas = [f for f in (fila.fecha for fila in planilla.registros) if f]
    if len(fechas) != len(set(fechas)):
        anomalies.append("Fechas duplicadas en registros (excluyendo vacías).")

    # Quick numeric sanity: large outliers compared to median-like heuristic.
    valores = []
    for fila in planilla.registros:
        for kg in (fila.kg_organicos, fila.kg_reciclaje, fila.kg_no_aprovechables):
            if kg is not None:
                valores.append(kg)
    if valores:
        media = sum(valores) / len(valores)
        for fila in planilla.registros:
            for label, kg in (
                ("organicos", fila.kg_organicos),
                ("reciclaje", fila.kg_reciclaje),
                ("no_aprovechables", fila.kg_no_aprovechables),
            ):
                if kg is not None and kg > max(media * 4, 50):
                    anomalies.append(
                        f"Valor atípico {kg} kg en {fila.fecha} ({label}), media aprox {media:.2f}"
                    )
    return anomalies


# -------------------------
# Consensus utilities
# -------------------------

def _choose_value(values: Dict[str, Any], priority_order: List[str]) -> Any:
    # Majority vote; if tie, use first in priority_order.
    counts: Dict[Any, int] = {}
    for v in values.values():
        counts[v] = counts.get(v, 0) + 1
    # Find max count values
    best_count = max(counts.values())
    best_values = [val for val, c in counts.items() if c == best_count]
    if len(best_values) == 1:
        return best_values[0]
    # Tie-breaker by priority: take value from first model in priority_order that appears in values
    for model in priority_order:
        if model in values:
            candidate = values[model]
            if candidate in best_values:
                return candidate
    return best_values[0]


def build_consensus(
    extractions: List[Dict[str, Any]], priority_order: List[str]
) -> Dict[str, Any]:
    """
    extractions: list of {"model": str, "planilla": PlanillaDigitalizada}
    returns {"planilla": PlanillaDigitalizada, "discrepancies": List[Dict]}
    """
    if not extractions:
        raise ValueError("No hay extracciones para consensuar")
    base = extractions[0]["planilla"]
    registros_consensus: List[FilaResiduo] = []
    discrepancies: List[Dict[str, Any]] = []

    max_rows = max(len(item["planilla"].registros) for item in extractions)
    for idx in range(max_rows):
        field_values: Dict[str, Dict[str, Any]] = {}
        for item in extractions:
            model = item["model"]
            regs = item["planilla"].registros
            if idx < len(regs):
                fila = regs[idx]
                field_values.setdefault("semana", {})[model] = fila.semana
                field_values.setdefault("dia", {})[model] = fila.dia
                field_values.setdefault("fecha", {})[model] = fila.fecha
                field_values.setdefault("kg_organicos", {})[model] = fila.kg_organicos
                field_values.setdefault("kg_reciclaje", {})[model] = fila.kg_reciclaje
                field_values.setdefault("kg_no_aprovechables", {})[model] = fila.kg_no_aprovechables
                field_values.setdefault("notas", {})[model] = fila.notas
        chosen = {}
        for campo, vals in field_values.items():
            chosen[campo] = _choose_value(vals, priority_order)
            # If not all equal, record discrepancy
            if len(set(vals.values())) > 1:
                discrepancies.append(
                    {
                        "fila_index": idx,
                        "campo": campo,
                        "valores": vals,
                        "valor_final": chosen[campo],
                    }
                )
        # Default placeholders if missing row in some extractions
        registros_consensus.append(
            FilaResiduo(
                semana=chosen.get("semana", f"Semana?"),
                dia=chosen.get("dia", "Desconocido"),
                fecha=chosen.get("fecha", "00/00/0000"),
                kg_organicos=chosen.get("kg_organicos"),
                kg_reciclaje=chosen.get("kg_reciclaje"),
                kg_no_aprovechables=chosen.get("kg_no_aprovechables"),
                notas=chosen.get("notas"),
            )
        )

    final_planilla = PlanillaDigitalizada(
        mes=base.mes,
        anio=base.anio,
        registros=registros_consensus,
        observaciones_generales=base.observaciones_generales,
        confianza_extraccion=base.confianza_extraccion,
    )
    return {"planilla": final_planilla, "discrepancies": discrepancies}


# -------------------------
# LangGraph nodes
# -------------------------

def build_graph(config: PipelineConfig, client: genai.Client):
    graph = StateGraph(PipelineState)

    async def ingest_node(state: PipelineState) -> PipelineState:
        if "image_bytes" in state:
            state["mime_type"] = state.get("mime_type", "image/jpeg")
            return state
        image_path = Path(state["image_path"])
        state["image_bytes"] = image_path.read_bytes()
        state["mime_type"] = state.get("mime_type", infer_mime_type(image_path))
        return state

    async def vision_extraction_node(state: PipelineState) -> PipelineState:
        attempt = state.get("attempt", 0) + 1
        enhance = state.get("enhance_contrast", False) or attempt > 1
        LOGGER.info("Ejecutando extracción (intento %s, contraste=%s)", attempt, enhance)
        models_to_try: List[str] = []
        for m in config.consensus_models:
            if not m:
                continue
            if "flash-image" in m or "image" in m:
                LOGGER.warning("Omitiendo modelo %s (no soporta JSON/schema)", m)
                continue
            if "tts" in m or "native-audio" in m:
                LOGGER.warning("Omitiendo modelo %s (modo TTS/audio, sin schema)", m)
                continue
            models_to_try.append(m)
        extractions: List[Dict[str, Any]] = []
        for model in models_to_try:
            per_model_attempts = 2
            for sub_try in range(per_model_attempts):
                try:
                    planilla = await parse_planilla_with_gemini(
                        client,
                        model,
                        state["image_bytes"],
                        mime_type=state.get("mime_type", "image/jpeg"),
                        enhance_contrast=enhance,
                    )
                    if planilla is None:
                        LOGGER.warning("Modelo %s devolvió resultado vacío", model)
                    else:
                        LOGGER.info("Extracción exitosa con modelo %s", model)
                        extractions.append({"model": model, "planilla": planilla})
                        break
                except (errors.ClientError, errors.ServerError) as exc:
                    LOGGER.warning(
                        "Fallo con modelo %s (int %s/%s): %s", model, sub_try + 1, per_model_attempts, exc
                    )
                    if sub_try + 1 >= per_model_attempts:
                        break
                    await asyncio.sleep(3)
        if not extractions:
            raise RuntimeError("No se pudo extraer la planilla con los modelos disponibles.")
        return {
            **state,
            "planilla": extractions[0]["planilla"],
            "extractions": extractions,
            "attempt": attempt,
            "needs_review": False,
            "enhance_contrast": enhance,
            "used_model": extractions[0]["model"],
        }

    async def consensus_node(state: PipelineState) -> PipelineState:
        extractions = [e for e in state.get("extractions", []) if e.get("planilla") is not None]
        priority = [e["model"] for e in extractions]
        result = build_consensus(extractions, priority_order=priority)
        return {
            **state,
            "planilla": result["planilla"],
            "planilla_consensus": result["planilla"],
            "discrepancies": result["discrepancies"],
        }

    async def sanity_check_node(state: PipelineState) -> PipelineState:
        planilla = state["planilla"]
        anomalies = detect_anomalies(planilla)
        audit_notes: List[str] = state.get("audit_notes", [])
        if anomalies:
            summary = await summarize_anomalies_with_flash(client, config.model_flash, planilla, anomalies)
            audit_notes.append(summary)
        needs_review = bool(anomalies)
        review_reason = "; ".join(anomalies) if anomalies else ""
        if needs_review and state.get("attempt", 1) < 2:
            LOGGER.info("Se detectaron anomalías. Reintentando con mejora de contraste.")
            return {
                **state,
                "needs_review": True,
                "review_reason": review_reason,
                "enhance_contrast": True,
                "audit_notes": audit_notes,
            }
        return {**state, "needs_review": needs_review, "review_reason": review_reason, "audit_notes": audit_notes}

    async def human_review_node(state: PipelineState) -> PipelineState:
        config.revision_dir.mkdir(parents=True, exist_ok=True)
        image_path = Path(state["image_path"])
        target = config.revision_dir / image_path.name
        target.write_bytes(state["image_bytes"])
        LOGGER.warning("Planilla marcada para revisión manual: %s", target)
        return {**state, "excel_path": "", "needs_review": True}

    async def export_node(state: PipelineState) -> PipelineState:
        planilla = state["planilla"]
        config.output_dir.mkdir(parents=True, exist_ok=True)
        excel_path = config.output_dir / f"planilla_{planilla.mes}_{planilla.anio}.xlsx"

        # Map de discrepancias por fila/campo
        discrepancy_map: Dict[int, set] = {}
        for d in state.get("discrepancies", []):
            idx = d.get("fila_index")
            campo = d.get("campo")
            if isinstance(idx, int) and campo:
                discrepancy_map.setdefault(idx, set()).add(campo)

        rows = []
        highlighted: List[tuple] = []  # (row_idx, col_name)
        for i, fila in enumerate(planilla.registros):
            fields = {
                "semana": fila.semana,
                "dia": fila.dia,
                "fecha": fila.fecha,
                "kg_organicos": fila.kg_organicos,
                "kg_reciclaje": fila.kg_reciclaje,
                "kg_no_aprovechables": fila.kg_no_aprovechables,
                "notas": fila.notas,
            }
            rows.append(fields)
            for campo in discrepancy_map.get(i, set()):
                highlighted.append((len(rows) - 1, campo))

        df = pd.DataFrame(rows)
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="registros")
            pd.DataFrame(
                [
                    {
                        "mes": planilla.mes,
                        "anio": planilla.anio,
                        "observaciones_generales": planilla.observaciones_generales,
                        "confianza_extraccion": planilla.confianza_extraccion,
                        "auditoria": "\n".join(state.get("audit_notes", [])),
                    }
                ]
            ).to_excel(writer, index=False, sheet_name="resumen")
            discrepancies = state.get("discrepancies", [])
            if discrepancies:
                pd.DataFrame(
                    [
                        {
                            "fila": d["fila_index"],
                            "campo": d["campo"],
                            "valor_final": d["valor_final"],
                            "valores_modelos": d["valores"],
                        }
                        for d in discrepancies
                    ]
                ).to_excel(writer, index=False, sheet_name="inconsistencias")
            # Resaltar celdas con discrepancias
            if highlighted:
                workbook = writer.book
                worksheet = writer.sheets["registros"]
                fmt = workbook.add_format({"bg_color": "#FFF3CD"})
                col_indices = {name: idx for idx, name in enumerate(df.columns)}
                for row_idx, col_name in highlighted:
                    col_idx = col_indices.get(col_name)
                    if col_idx is None:
                        continue
                    value = df.iloc[row_idx, col_idx]
                    worksheet.write(row_idx + 1, col_idx, value, fmt)
        LOGGER.info("Exportado a Excel en %s", excel_path)
        return {**state, "excel_path": str(excel_path)}

    graph.add_node("Ingesta", ingest_node)
    graph.add_node("VisionExtraction", vision_extraction_node)
    graph.add_node("Consensus", consensus_node)
    graph.add_node("SanityCheck", sanity_check_node)
    graph.add_node("HumanReview", human_review_node)
    graph.add_node("Export", export_node)

    graph.set_entry_point("Ingesta")
    graph.add_edge("Ingesta", "VisionExtraction")
    graph.add_edge("VisionExtraction", "Consensus")
    graph.add_edge("Consensus", "SanityCheck")

    def route_after_sanity(state: PipelineState) -> str:
        if state.get("needs_review"):
            if state.get("attempt", 1) < 2:
                return "VisionExtraction"
            # Después del reintento, exportamos igualmente pero marcamos discrepancias.
            return "Export"
        return "Export"

    graph.add_conditional_edges(
        "SanityCheck",
        route_after_sanity,
        {
            "VisionExtraction": "VisionExtraction",
            "HumanReview": "HumanReview",
            "Export": "Export",
        },
    )
    graph.add_edge("HumanReview", END)
    graph.add_edge("Export", END)

    return graph.compile()


# -------------------------
# Public API
# -------------------------

async def run_pipeline(image_path: str, api_key: Optional[str] = None) -> PipelineState:
    if Path(image_path).suffix.lower() == ".pdf":
        result = await run_batch([image_path], api_key=api_key)
        return {"excel_path": result.get("excel_path", ""), "processed": result.get("processed", 0)}
    client = build_client(api_key)
    config = PipelineConfig()
    app = build_graph(config, client)
    initial_state: PipelineState = {"image_path": image_path, "attempt": 0}
    result: Dict[str, Any] = await app.ainvoke(initial_state)  # type: ignore[assignment]
    return result  # Already a PipelineState


def encode_example_image(path: str) -> str:
    """Helper to turn an image into base64 for quick debugging."""
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def infer_mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".png":
        return "image/png"
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "image/jpeg"


def render_pdf_to_jpeg_bytes(pdf_path: Path, zoom: float = 2.0) -> List[bytes]:
    images: List[bytes] = []
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            images.append(pix.tobytes("jpeg"))
    return images


def build_input_items(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".pdf":
        pages = render_pdf_to_jpeg_bytes(path)
        return [
            {
                "label": f"{path.name}#page-{idx}",
                "image_bytes": image_bytes,
                "mime_type": "image/jpeg",
            }
            for idx, image_bytes in enumerate(pages, start=1)
        ]
    return [
        {
            "label": path.name,
            "image_path": str(path),
            "mime_type": infer_mime_type(path),
        }
    ]


def list_image_files(dir_path: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".pdf"}
    paths = [
        str(p)
        for p in sorted(Path(dir_path).iterdir())
        if p.is_file() and p.suffix.lower() in exts
    ]
    return paths


async def run_batch(image_paths: List[str], api_key: Optional[str] = None) -> Dict[str, Any]:
    """Procesa múltiples archivos (imágenes/PDF) y genera un Excel consolidado."""
    if not image_paths:
        raise ValueError("No se proporcionaron archivos para el batch.")
    client = build_client(api_key)
    config = PipelineConfig()
    app = build_graph(config, client)

    combined_rows: List[Dict[str, Any]] = []
    resumen_rows: List[Dict[str, Any]] = []
    inconsist_rows: List[Dict[str, Any]] = []
    highlighted_cells: List[tuple] = []  # (global_row_idx, col_name)
    processed = 0

    for img in image_paths:
        input_path = Path(img)
        try:
            items = build_input_items(input_path)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.error("No se pudo preparar %s: %s", img, exc)
            continue
        if not items:
            LOGGER.warning("Archivo sin páginas útiles: %s", img)
            continue
        for item in items:
            label = item["label"]
            LOGGER.info("Procesando archivo: %s", label)
            try:
                state_input: Dict[str, Any] = {
                    "image_path": item.get("image_path", label),
                    "attempt": 0,
                    "mime_type": item.get("mime_type", "image/jpeg"),
                }
                if "image_bytes" in item:
                    state_input["image_bytes"] = item["image_bytes"]
                state = await app.ainvoke(state_input)  # type: ignore[assignment]
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.error("Error procesando %s: %s", label, exc)
                continue
            planilla: Optional[PlanillaDigitalizada] = state.get("planilla")
            if not planilla:
                LOGGER.warning("Sin planilla para %s; se omite en consolidado.", label)
                continue

        # Marca de discrepancias
            discrepancy_map: Dict[int, set] = {}
            for d in state.get("discrepancies", []):
                idx = d.get("fila_index")
                campo = d.get("campo")
                if isinstance(idx, int) and campo:
                    discrepancy_map.setdefault(idx, set()).add(campo)

            def mark(val: Any, flagged: bool) -> Any:
                return val

            start_idx = len(combined_rows)
            for i, fila in enumerate(planilla.registros):
                combined_rows.append(
                    {
                        "archivo": label,
                        "mes": planilla.mes,
                        "anio": planilla.anio,
                        "semana": mark(fila.semana, "semana" in discrepancy_map.get(i, set())),
                        "dia": mark(fila.dia, "dia" in discrepancy_map.get(i, set())),
                        "fecha": mark(fila.fecha, "fecha" in discrepancy_map.get(i, set())),
                        "kg_organicos": mark(fila.kg_organicos, "kg_organicos" in discrepancy_map.get(i, set())),
                        "kg_reciclaje": mark(fila.kg_reciclaje, "kg_reciclaje" in discrepancy_map.get(i, set())),
                        "kg_no_aprovechables": mark(
                            fila.kg_no_aprovechables, "kg_no_aprovechables" in discrepancy_map.get(i, set())
                        ),
                        "notas": mark(fila.notas, "notas" in discrepancy_map.get(i, set())),
                        "confianza_extraccion": planilla.confianza_extraccion,
                    }
                )
                for campo in discrepancy_map.get(i, set()):
                    highlighted_cells.append((start_idx + i, campo))

            resumen_rows.append(
                {
                    "archivo": label,
                    "mes": planilla.mes,
                    "anio": planilla.anio,
                    "observaciones_generales": planilla.observaciones_generales,
                    "confianza_extraccion": planilla.confianza_extraccion,
                    "auditoria": "\n".join(state.get("audit_notes", [])),
                    "excel_individual": state.get("excel_path", ""),
                    "needs_review": state.get("needs_review", False),
                }
            )

            for d in state.get("discrepancies", []):
                inconsist_rows.append(
                    {
                        "archivo": label,
                        "fila": d.get("fila_index"),
                        "campo": d.get("campo"),
                        "valor_final": d.get("valor_final"),
                        "valores_modelos": d.get("valores"),
                    }
                )
            processed += 1

    if not combined_rows:
        LOGGER.error("No se generó ningún registro consolidado.")
        return {"excel_path": "", "processed": processed}

    config.output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = config.output_dir / "planillas_consolidadas.xlsx"
    with pd.ExcelWriter(combined_path, engine="xlsxwriter") as writer:
        pd.DataFrame(combined_rows).to_excel(writer, index=False, sheet_name="registros")
        pd.DataFrame(resumen_rows).to_excel(writer, index=False, sheet_name="resumen")
        if inconsist_rows:
            pd.DataFrame(inconsist_rows).to_excel(writer, index=False, sheet_name="inconsistencias")
        # Resaltar discrepancias
        if highlighted_cells:
            df = pd.DataFrame(combined_rows)
            worksheet = writer.sheets["registros"]
            fmt = writer.book.add_format({"bg_color": "#FFF3CD"})
            col_indices = {name: idx for idx, name in enumerate(df.columns)}
            for row_idx, col_name in highlighted_cells:
                col_idx = col_indices.get(col_name)
                if col_idx is None:
                    continue
                value = df.iloc[row_idx, col_idx]
                worksheet.write(row_idx + 1, col_idx, value, fmt)

    LOGGER.info("Exportado consolidado a Excel en %s", combined_path)
    return {"excel_path": str(combined_path), "processed": processed}


def cli():
    import argparse

    parser = argparse.ArgumentParser(description="Ingesta de planillas con Gemini 3.0")
    parser.add_argument("image", nargs="?", help="Ruta a la imagen o PDF de la planilla (jpg/png/pdf).")
    parser.add_argument(
        "--dir",
        dest="dir",
        help="Ruta a carpeta con archivos (jpg/png/pdf) para procesar en batch.",
    )
    parser.add_argument("--api-key", dest="api_key", help="API key de Google (opcional, usa GOOGLE_API_KEY).")
    parser.add_argument(
        "--model-pro",
        dest="model_pro",
        help="Override modelo pro (default env GENAI_PRO_MODEL o gemini-3.0-pro-001).",
    )
    parser.add_argument(
        "--model-flash",
        dest="model_flash",
        help="Override modelo flash (default env GENAI_FLASH_MODEL o gemini-3.0-flash-001).",
    )
    parser.add_argument(
        "--model-pro-fallbacks",
        dest="model_pro_fallbacks",
        help="Lista separada por comas de fallbacks para el modelo pro (ej: models/gemini-pro-latest,models/gemini-2.5-flash).",
    )
    args = parser.parse_args()

    async def _main():
        try:
            # Allow runtime model overrides
            if args.model_pro or args.model_flash:
                os.environ.setdefault("GENAI_PRO_MODEL", args.model_pro or "")
                os.environ.setdefault("GENAI_FLASH_MODEL", args.model_flash or "")
            if args.model_pro_fallbacks:
                os.environ["GENAI_PRO_FALLBACKS"] = args.model_pro_fallbacks
            # Directorio explícito
            if args.dir:
                files = list_image_files(args.dir)
                if not files:
                    LOGGER.error("No se encontraron archivos (jpg/png/pdf) en %s", args.dir)
                    return
                result = await run_batch(files, api_key=args.api_key)
                if result.get("excel_path"):
                    LOGGER.info("Batch completado. Excel consolidado: %s", result["excel_path"])
                else:
                    LOGGER.error("Batch finalizado sin exportar.")
                return
            # Si el argumento principal es un directorio, trata como batch
            if args.image and Path(args.image).is_dir():
                files = list_image_files(args.image)
                if not files:
                    LOGGER.error("No se encontraron archivos (jpg/png/pdf) en %s", args.image)
                    return
                result = await run_batch(files, api_key=args.api_key)
                if result.get("excel_path"):
                    LOGGER.info("Batch completado. Excel consolidado: %s", result["excel_path"])
                else:
                    LOGGER.error("Batch finalizado sin exportar.")
                return
            if not args.image:
                LOGGER.error("Debes proporcionar una imagen/PDF o un directorio.")
                return
            state = await run_pipeline(args.image, api_key=args.api_key)
        except (ValidationError, RuntimeError) as exc:
            LOGGER.error("Error en la ingesta: %s", exc)
            return
        if state.get("excel_path"):
            LOGGER.info("Pipeline completado. Excel: %s", state["excel_path"])
        elif state.get("needs_review"):
            LOGGER.warning("Planilla enviada a revisión: %s", state.get("review_reason", ""))
        else:
            LOGGER.error("Pipeline finalizado sin exportar ni marcar revisión.")

    asyncio.run(_main())


if __name__ == "__main__":
    cli()
