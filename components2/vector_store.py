from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

def create_vectorstore(chunks):
    try:
        embeddings = GPT4AllEmbeddings(model_file=r"D:\Doan\askpdf\models\all-MiniLM-L6-v2-f16 .gguf")
        return FAISS.from_texts(texts=chunks, embedding=embeddings)
    except Exception as e:
        return f"Lỗi khi tạo vector store: {str(e)}"