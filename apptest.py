# import streamlit as st
# from PyPDF2 import PdfReader
# from transformers import AutoTokenizer, AutoModel, pipeline
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings  # Changed import
# import torch
#
#
# class VietnameseLLM:
#     def __init__(self, model_name: str = "NlpHUST/gpt2-vietnamese"):
#         self.generator = pipeline(
#             "text-generation",
#             model=model_name,
#             tokenizer=model_name
#         )
#
#     def generate(self, prompt: str) -> str:
#         try:
#             # Use max_new_tokens instead of max_length
#             response = self.generator(
#                 prompt,
#                 max_new_tokens=100,  # Maximum number of new tokens to generate
#                 num_return_sequences=1,
#                 pad_token_id=self.generator.tokenizer.eos_token_id,
#                 do_sample=True,
#                 temperature=0.7,
#                 top_k=50,
#                 top_p=0.95,
#                 no_repeat_ngram_size=3
#             )
#
#             # Clean up the generated text
#             generated_text = response[0]['generated_text']
#             if generated_text.startswith(prompt):
#                 generated_text = generated_text[len(prompt):].strip()
#
#             return generated_text
#         except Exception as e:
#             return f"Lỗi khi sinh văn bản: {str(e)}"
#
#
# def read_pdf(file_path):
#     try:
#         pdf_reader = PdfReader(file_path)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#         return text
#     except Exception as e:
#         st.error(f"Lỗi khi đọc PDF: {str(e)}")
#         return ""
#
#
# def chunk_text(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=500,
#         chunk_overlap=50,
#         length_function=len
#     )
#     return text_splitter.split_text(text)
#
#
# def create_vectorstore(chunks):
#     try:
#         embeddings = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-MiniLM-L6-v2",
#             model_kwargs={'device': 'cpu'}
#         )
#         return FAISS.from_texts(texts=chunks, embedding=embeddings)
#     except Exception as e:
#         st.error(f"Lỗi khi tạo vector store: {str(e)}")
#         return None
#
#
# def chatbot_response(user_input: str, vectorstore, llm: VietnameseLLM) -> str:
#     try:
#         if vectorstore is None:
#             return "Chưa có dữ liệu để trả lời."
#
#         retriever = vectorstore.as_retriever(
#             search_type="similarity",
#             search_kwargs={"k": 2}
#         )
#         docs = retriever.get_relevant_documents(user_input)
#
#         if not docs:
#             return "Không tìm thấy thông tin liên quan trong tài liệu."
#
#         # Optimize prompt structure
#         context = "\n".join([doc.page_content for doc in docs])
#         prompt = f"""Dưới đây là nội dung từ tài liệu:
# {context}
#
# Câu hỏi: {user_input}
# Trả lời ngắn gọn và chính xác:"""
#
#         return llm.generate(prompt)
#     except Exception as e:
#         return f"Lỗi khi xử lý câu hỏi: {str(e)}"
#
#
# def main():
#     st.set_page_config(page_title="Chat với PDF Tiếng Việt", layout="wide")
#     st.title("Chat với PDF Tiếng Việt")
#
#     # Initialize session state
#     if 'llm' not in st.session_state:
#         with st.spinner('Đang khởi tạo mô hình...'):
#             st.session_state.llm = VietnameseLLM()
#
#     if 'vectorstore' not in st.session_state:
#         st.session_state.vectorstore = None
#
#     # Sidebar for PDF upload
#     with st.sidebar:
#         st.subheader("Tải lên file PDF")
#         pdf_file = st.file_uploader("Chọn file PDF", type="pdf")
#
#         if pdf_file:
#             with st.spinner('Đang xử lý file PDF...'):
#                 context = read_pdf(pdf_file)
#                 if context:
#                     chunks = chunk_text(context)
#                     st.session_state.vectorstore = create_vectorstore(chunks)
#                     if st.session_state.vectorstore:
#                         st.success("Đã xử lý file PDF thành công!")
#
#     # Main chat interface
#     st.subheader("Đặt câu hỏi về nội dung PDF:")
#
#     # Initialize chat history
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []
#
#     # Display chat history
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
#
#     # Chat input
#     if prompt := st.chat_input("Nhập câu hỏi của bạn"):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)
#
#         if st.session_state.vectorstore is None:
#             response = "Vui lòng tải lên file PDF trước khi đặt câu hỏi."
#         else:
#             with st.chat_message("assistant"):
#                 with st.spinner('Đang xử lý câu hỏi...'):
#                     response = chatbot_response(
#                         prompt,
#                         st.session_state.vectorstore,
#                         st.session_state.llm
#                     )
#                 st.markdown(response)
#
#         st.session_state.messages.append({"role": "assistant", "content": response})
#
#
# if __name__ == "__main__":
#     main()
# import streamlit as st
# from PyPDF2 import PdfReader
# from transformers import pipeline
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
#
# class VietnameseLLM:
#     def __init__(self, model_name: str = "NlpHUST/gpt2-vietnamese"):
#         self.generator = pipeline(
#             "text-generation",
#             model=model_name,
#             tokenizer=model_name
#         )
#
#     def generate(self, prompt: str) -> str:
#         try:
#             # Kiểm tra xem prompt có rỗng hay không
#             if len(prompt.strip()) == 0:
#                 return "Vui lòng cung cấp câu hỏi hoặc yêu cầu hợp lệ."
#
#             response = self.generator(
#                 prompt,
#                 max_new_tokens=150,
#                 num_return_sequences=1,
#                 pad_token_id=self.generator.tokenizer.eos_token_id,
#                 do_sample=True,
#                 temperature=0.5,
#                 top_k=50,
#                 top_p=0.95,
#                 no_repeat_ngram_size=3
#             )
#             generated_text = response[0]['generated_text']
#
#             # Kiểm tra nếu kết quả sinh ra quá ngắn so với prompt
#             if len(generated_text) <= len(prompt):
#                 return "Không thể sinh câu trả lời, vui lòng thử lại."
#
#             # Trả về phần văn bản mới sinh ra sau phần prompt
#             return generated_text[len(prompt):].strip() if generated_text.startswith(prompt) else generated_text.strip()
#
#         except Exception as e:
#             return f"Lỗi khi sinh văn bản: {str(e)}"
#
# def read_pdf(file_path):
#     try:
#         pdf_reader = PdfReader(file_path)
#         return "".join(page.extract_text() for page in pdf_reader.pages)
#     except Exception as e:
#         st.error(f"Lỗi khi đọc PDF: {str(e)}")
#         return ""
#
# def chunk_text(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=500,
#         chunk_overlap=50,
#         length_function=len
#     )
#     return text_splitter.split_text(text)
#
# def create_vectorstore(chunks):
#     try:
#         embeddings = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-MiniLM-L6-v2",
#             model_kwargs={'device': 'cpu'}
#         )
#         return FAISS.from_texts(texts=chunks, embedding=embeddings)
#     except Exception as e:
#         st.error(f"Lỗi khi tạo vector store: {str(e)}")
#         return None
#
# def chatbot_response(user_input: str, vectorstore, llm: VietnameseLLM) -> str:
#     try:
#         if vectorstore is None:
#             return "Chưa có dữ liệu để trả lời."
#
#         retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
#         docs = retriever.get_relevant_documents(user_input)
#
#         if not docs:
#             return "Không tìm thấy thông tin liên quan trong tài liệu."
#
#         context = "\n".join([doc.page_content for doc in docs])
#         prompt = f"""Dưới đây là nội dung từ tài liệu:
# {context}
#
# Câu hỏi: {user_input}
# Trả lời ngắn gọn và chính xác:"""
#
#         return llm.generate(prompt)
#     except Exception as e:
#         return f"Lỗi khi xử lý câu hỏi: {str(e)}"
#
# def main():
#     st.set_page_config(page_title="ChatBot", layout="wide")
#     st.title("Chào mừng bạn tới với chatbot")
#
#     if 'llm' not in st.session_state:
#         with st.spinner('Đang khởi tạo mô hình...'):
#             st.session_state.llm = VietnameseLLM()
#
#     if 'vectorstore' not in st.session_state:
#         st.session_state.vectorstore = None
#
#     with st.sidebar:
#         st.subheader("Tải lên file PDF")
#         pdf_file = st.file_uploader("Chọn file PDF", type="pdf")
#
#         if pdf_file:
#             with st.spinner('Đang xử lý file PDF...'):
#                 context = read_pdf(pdf_file)
#                 if context:
#                     chunks = chunk_text(context)
#                     st.session_state.vectorstore = create_vectorstore(chunks)
#                     if st.session_state.vectorstore:
#                         st.success("Đã xử lý file PDF thành công!")
#
#     st.subheader("Đặt câu hỏi về nội dung PDF:")
#
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []
#
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
#
#     if prompt := st.chat_input("Nhập câu hỏi của bạn"):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)
#
#         if st.session_state.vectorstore is None:
#             response = "Vui lòng tải lên file PDF trước khi đặt câu hỏi."
#         else:
#             with st.chat_message("assistant"):
#                 with st.spinner('Đang xử lý câu hỏi...'):
#                     response = chatbot_response(prompt, st.session_state.vectorstore, st.session_state.llm)
#                 st.markdown(response)
#
#         st.session_state.messages.append({"role": "assistant", "content": response})
#
# if __name__ == "__main__":
#     main()
# _______________________________________________________________________________________________________