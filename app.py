import asyncio
import time
import tempfile
from pathlib import Path
from typing import List

import streamlit as st

from main import run_batch, list_image_files


def save_uploads(uploaded_files) -> List[str]:
    """Guarda los archivos subidos a un directorio temporal y devuelve rutas."""
    temp_dir = Path(tempfile.mkdtemp(prefix="planillas_"))
    paths: List[str] = []
    for f in uploaded_files:
        suffix = Path(f.name).suffix or ".jpg"
        target = temp_dir / Path(f.name).name
        target.write_bytes(f.read())
        paths.append(str(target))
    return paths


def clear_previous_result() -> None:
    for key in (
        "last_excel_bytes",
        "last_excel_name",
        "last_processed",
        "last_elapsed",
        "last_error",
    ):
        st.session_state.pop(key, None)


def format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"


def main():
    st.set_page_config(page_title="Ingesta de Planillas", page_icon="🗂️", layout="centered")
    st.title("Ingesta de Planillas de Basuras")
    st.write(
        "Sube imágenes (jpg/png) o PDFs de las planillas manuscritas. "
        "El sistema ejecuta el pipeline con consenso multi-modelo y exporta un Excel consolidado."
    )

    api_key = st.text_input("API Key de Gemini", type="password")
    model_pro = st.text_input("Modelo Pro (thinking)", value="models/gemini-3-pro-preview")
    model_flash = st.text_input("Modelo Flash (rápido)", value="models/gemini-flash-latest")

    uploaded = st.file_uploader(
        "Sube una o varias imágenes o PDFs",
        type=["jpg", "jpeg", "png", "pdf"],
        accept_multiple_files=True,
        key="uploads",
        on_change=clear_previous_result,
    )

    run_button = st.button("Procesar")

    if run_button:
        clear_previous_result()
        if not api_key:
            st.error("Debes ingresar la API Key.")
            return
        if not uploaded:
            st.error("Debes subir al menos una imagen o PDF.")
            return

        with st.spinner("Procesando planillas..."):
            start = time.perf_counter()
            paths = save_uploads(uploaded)
            # Opcional: permitir leer carpeta entera (por si se usa como entrada)
            # paths = list_image_files(<dir>) si se prefiere.
            result = asyncio.run(run_batch(paths, api_key=api_key))
            elapsed = time.perf_counter() - start

        if result.get("excel_path"):
            excel_path = Path(result["excel_path"])
            st.session_state["last_excel_bytes"] = excel_path.read_bytes()
            st.session_state["last_excel_name"] = excel_path.name
            st.session_state["last_processed"] = result.get("processed", 0)
            st.session_state["last_elapsed"] = elapsed
        else:
            st.session_state["last_error"] = "No se generó el Excel consolidado. Revisa los logs para más detalle."

    excel_bytes = st.session_state.get("last_excel_bytes")
    if excel_bytes:
        processed = st.session_state.get("last_processed", 0)
        elapsed = st.session_state.get("last_elapsed")
        st.success(f"Batch completado. Archivos procesados: {processed}")
        if elapsed is not None:
            st.write(f"Tiempo total de procesamiento: {format_elapsed(elapsed)}")
        st.download_button(
            "Descargar Excel consolidado",
            data=excel_bytes,
            file_name=st.session_state.get("last_excel_name", "planillas_consolidadas.xlsx"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel",
        )
    elif st.session_state.get("last_error"):
        st.error(st.session_state["last_error"])


if __name__ == "__main__":
    main()
