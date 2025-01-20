import streamlit as st
from components2.pdf_processing import read_pdf, chunk_text
from components2.vector_store import create_vectorstore
from components2.llm import CustomLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Cấu hình Streamlit
st.set_page_config(page_title="ChatBot", layout="wide")

# Tao prompt template
def creat_prompt(template):
    prompt = PromptTemplate(template = template, input_variables=["context", "question"])
    return prompt


# Tao qa chain
def create_qa_chain(prompt, llm, vectordb):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        retriever = vectordb.as_retriever(search_kwargs = {"k":3}, max_tokens_limit=1024),
        return_source_documents = False,
        chain_type_kwargs= {'prompt': prompt}

    )
    return llm_chain

#Tao Prompt
template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""

prompt = creat_prompt(template)

# Khởi tạo LLM trong session state
if 'llm' not in st.session_state:
    with st.spinner('Đang khởi tạo mô hình...'):
        model_file = r"D:\Doan\askpdf\models\vinallama-7b-chat_q5_0.gguf"
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

if query := st.chat_input("Nhập câu hỏi của bạn"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.vectorstore is None:
        response = "Vui lòng tải lên file PDF trước khi đặt câu hỏi."
    else:
        with st.chat_message("assistant"):
            with st.spinner('Đang xử lý câu hỏi...'):
                vectordb = st.session_state.vectorstore.as_retriever(search_kwargs={'k': 3})
                llm_chain = create_qa_chain(prompt, st.session_state.llm, vectordb)
                response = llm_chain.invoke({"query": query})
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
