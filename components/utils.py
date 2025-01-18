import streamlit as st
from PIL import Image

def show_uploaded_image(image_file):
    """Hiển thị ảnh đã tải lên"""
    try:
        image = Image.open(image_file)
        st.image(image, caption='Ảnh đã tải lên', use_column_width=True)
    except Exception as e:
        st.error(f"Lỗi khi hiển thị ảnh: {str(e)}")