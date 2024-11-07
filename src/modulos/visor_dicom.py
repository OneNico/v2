# src/modulos/visor_dicom.py

import streamlit as st
from PIL import Image, ImageEnhance
import pydicom
import numpy as np
from io import BytesIO
import logging
from pydicom.pixels import apply_voi_lut
import base64

from src.modulos.procesamiento_i import procesamiento_individual
# El Detector será pasado como parámetro, no es necesario importarlo aquí

logger = logging.getLogger(__name__)

def visualizar_dicom(opciones, detector):
    st.write("---")
    st.header("Visor Avanzado de Imágenes DICOM")

    # Subir múltiples archivos DICOM
    uploaded_files = st.file_uploader(
        "Cargar archivos DICOM",
        type=["dcm", "dicom"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Por favor, carga uno o más archivos DICOM para visualizar.")
        return

    num_imagenes = len(uploaded_files)
    logger.info(f"Cantidad de imágenes cargadas para visualización: {num_imagenes}")

    # Seleccionar la imagen a visualizar
    selected_file = st.selectbox(
        "Selecciona una imagen DICOM para visualizar",
        uploaded_files,
        format_func=lambda x: x.name
    )

    # Checkbox para abrir el visor
    abrir_visor = st.checkbox("Abrir Visor DICOM")

    if abrir_visor:
        mostrar_visor(selected_file, opciones, detector)

def mostrar_visor(selected_file, opciones, detector):
    imagen, ds, pixel_spacing = procesar_imagen_dicom(selected_file)

    if imagen is not None:
        # Control de brillo y contraste
        col1, col2 = st.columns([1, 1])
        with col1:
            brillo = st.slider("Brillo", -100, 100, 0, key=f"brillo_{selected_file.name}")
        with col2:
            contraste = st.slider("Contraste", -100, 100, 0, key=f"contraste_{selected_file.name}")

        # Aplicar ajustes de brillo y contraste
        imagen_editada = ajustar_brillo_contraste(imagen, brillo, contraste)

        # Convertir la imagen a base64 para incrustarla en HTML
        buffered = BytesIO()
        imagen_editada.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # HTML y JavaScript para mostrar la imagen (omitido por brevedad)
        # ... [tu código HTML y JS para mostrar la imagen con zoom y arrastre] ...

        # Mostrar la imagen con funcionalidad de arrastre, zoom, rotación y reset
        st.components.v1.html(draggable_image_html, height=800)  # Ajustar la altura a 800

        # Opciones de Detección de Objetos
        if opciones.get('activar_deteccion', False):
            st.subheader("Detección de Objetos en la Imagen")

            detectar_objetos = st.button("Detectar Objetos", key=f"detectar_{selected_file.name}")

            if detectar_objetos:
                with st.spinner("Realizando detección de objetos..."):
                    imagen_detectada = detector.detectar_objetos(imagen_editada)
                    if imagen_detectada:
                        st.image(imagen_detectada, caption="Imagen con Detecciones", use_column_width=True)
                    else:
                        st.error("No se pudo realizar la detección de objetos.")
        else:
            st.info("La detección de objetos está desactivada. Actívala en la barra lateral para usar esta función.")

        # Descargar DICOM modificado y PNG de alta resolución, y botón "Analizar mamografía"
        st.subheader("Descargar Imagen Modificada")

        # Convertir la imagen editada a escala de grises para guardar en DICOM
        imagen_gris = imagen_editada.convert("L")
        pixel_data = np.array(imagen_gris)

        # Botones para descargar en DICOM, PNG y Analizar mamografía
        col1, col2, col3 = st.columns(3)
        with col1:
            dicom_buffer = BytesIO()
            try:
                # Aplicar los cambios al dataset DICOM
                ds.PixelData = pixel_data.tobytes()
                ds.Rows, ds.Columns = pixel_data.shape

                ds.save_as(dicom_buffer)
                dicom_buffer.seek(0)

                # Preparar el archivo para descarga
                st.download_button(
                    label="Descargar DICOM Modificado",
                    data=dicom_buffer,
                    file_name=f"modificado_{selected_file.name}",
                    mime="application/dicom",
                    key="download_dicom"
                )
            except Exception as e:
                st.error(f"Error al preparar el archivo DICOM: {e}")

        with col2:
            try:
                png_buffer = BytesIO()
                # Guardar la imagen editada en el buffer con alta calidad
                imagen_editada.save(png_buffer, format="PNG")
                png_buffer.seek(0)

                # Preparar el archivo para descarga
                st.download_button(
                    label="Descargar PNG de Alta Resolución",
                    data=png_buffer,
                    file_name=f"modificado_{selected_file.name}.png",
                    mime="image/png",
                    key="download_png"
                )
            except Exception as e:
                st.error(f"Error al preparar la imagen PNG: {e}")

        with col3:
            analizar = st.button("Analizar mamografía", key=f"analizar_{selected_file.name}")

        if analizar:
            with st.spinner("Analizando..."):
                opciones_procesamiento = {
                    'uploaded_image': selected_file
                }
                procesamiento_individual(opciones_procesamiento)
    else:
        st.error(f"No se pudo procesar la imagen {selected_file.name}")

def procesar_imagen_dicom(dicom_file):
    try:
        dicom = pydicom.dcmread(BytesIO(dicom_file.getvalue()))
        original_image = dicom.pixel_array

        pixel_spacing = dicom.get('PixelSpacing', None)
        if pixel_spacing is not None:
            pixel_spacing = [float(spacing) for spacing in pixel_spacing]
        else:
            pixel_spacing = dicom.get('ImagerPixelSpacing', None)
            if pixel_spacing is not None:
                pixel_spacing = [float(spacing) for spacing in pixel_spacing]
            else:
                pixel_spacing = [1, 1]

        img_windowed = apply_voi_lut(original_image, dicom)

        photometric_interpretation = dicom.get('PhotometricInterpretation', 'UNKNOWN')
        if photometric_interpretation == 'MONOCHROME1':
            img_windowed = np.max(img_windowed) - img_windowed

        img_normalized_display = (img_windowed - np.min(img_windowed)) / (np.max(img_windowed) - np.min(img_windowed))
        img_normalized_display = (img_normalized_display * 255).astype(np.uint8)

        image_display = Image.fromarray(img_normalized_display).convert('L')

        return image_display, dicom, pixel_spacing

    except Exception as e:
        logger.error(f"Error al procesar el archivo DICOM: {e}")
        st.error(f"Error al procesar el archivo DICOM: {e}")
        return None, None, None

def ajustar_brillo_contraste(imagen, brillo, contraste):
    try:
        enhancer = ImageEnhance.Brightness(imagen)
        imagen = enhancer.enhance(1 + brillo / 100)

        enhancer = ImageEnhance.Contrast(imagen)
        imagen = enhancer.enhance(1 + contraste / 100)

        return imagen
    except Exception as e:
        logger.error(f"Error al ajustar brillo y contraste: {e}")
        st.error(f"Error al ajustar brillo y contraste: {e}")
        return imagen
