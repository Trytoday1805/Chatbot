import streamlit as st
from components.pdf_processing import read_pdf, chunk_text
from components.vector_store import create_vectorstore
from components.llm import CustomLLM, chatbot_response

# Cấu hình Streamlit
st.set_page_config(page_title="ChatBot", layout="wide")

# Khởi tạo LLM trong session state
if 'llm' not in st.session_state:
    with st.spinner('Đang khởi tạo mô hình...'):
        model_file = r"D:\Nam 3\API\vinallama-7b-chat_q5_0.gguf"
        st.session_state.llm = CustomLLM(model_file)

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

st.markdown("<h1 style='text-align: center;'><i class='fa fa-comments'></i> Chào mừng bạn tới với Chatbot</h1>", unsafe_allow_html=True)


# Sidebar: Tải lên file PDF
with st.sidebar:
    st.subheader("Tải lên file PDF")
    pdf_file = st.file_uploader("Chọn file PDF", type="pdf")

    if pdf_file:
        with st.spinner('Đang xử lý file PDF...'):
            context = read_pdf(pdf_file)
            if context:
                chunks = chunk_text(context)
                st.session_state.vectorstore = create_vectorstore(chunks)
                if st.session_state.vectorstore:
                    st.success("Đã xử lý thành công!")

# Khối chatbot
st.subheader("Hãy đặt câu hỏi về nội dung PDF:")
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
