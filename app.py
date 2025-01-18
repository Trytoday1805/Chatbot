import streamlit as st
from components.pdf_processing import read_pdf, chunk_text
from components.vector_store import create_vectorstore
from components.llm import VietnameseLLM, chatbot_response
from components.utils import show_uploaded_image

# Đảm bảo gọi set_page_config đầu tiên
st.set_page_config(page_title="ChatBot", layout="wide")

# Tạo đối tượng LLM và vector store từ session_state
if 'llm' not in st.session_state:
    with st.spinner('Đang khởi tạo mô hình...'):
        st.session_state.llm = VietnameseLLM()

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

st.title("Chào mừng bạn tới với chatbot")

# Sidebar: Tải lên file PDF và hiển thị ảnh
with st.sidebar:
    st.subheader("Tải lên file PDF")
    pdf_file = st.file_uploader("Chọn file PDF", type="pdf")
    image_file = st.file_uploader("Chọn hình ảnh", type=["png", "jpg", "jpeg"])
    if image_file:
        show_uploaded_image(image_file)

    if pdf_file:
        with st.spinner('Đang xử lý file PDF...'):
            context = read_pdf(pdf_file)
            if context:
                chunks = chunk_text(context)
                st.session_state.vectorstore = create_vectorstore(chunks)
                if st.session_state.vectorstore:
                    st.success("Đã xử lý file PDF thành công!")

# Khối chatbot: Đặt câu hỏi
st.subheader("Đặt câu hỏi về nội dung PDF:")
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Nhập câu hỏi của bạn"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.vectorstore is None:
        response = "Vui lòng tải lên file PDF trước khi đặt câu hỏi."
    else:
        with st.chat_message("assistant"):
            with st.spinner('Đang xử lý câu hỏi...'):
                response = chatbot_response(prompt, st.session_state.vectorstore, st.session_state.llm)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
# _____________________
