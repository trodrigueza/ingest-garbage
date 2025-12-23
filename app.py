import asyncio
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
    )

    run_button = st.button("Procesar")

    if run_button:
        if not api_key:
            st.error("Debes ingresar la API Key.")
            return
        if not uploaded:
            st.error("Debes subir al menos una imagen o PDF.")
            return

        with st.spinner("Procesando planillas..."):
            paths = save_uploads(uploaded)
            # Opcional: permitir leer carpeta entera (por si se usa como entrada)
            # paths = list_image_files(<dir>) si se prefiere.
            result = asyncio.run(run_batch(paths, api_key=api_key))

        if result.get("excel_path"):
            excel_path = Path(result["excel_path"])
            st.success(f"Batch completado. Archivos procesados: {result.get('processed', 0)}")
            st.download_button(
                "Descargar Excel consolidado",
                data=excel_path.read_bytes(),
                file_name=excel_path.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.error("No se generó el Excel consolidado. Revisa los logs para más detalle.")


if __name__ == "__main__":
    main()
