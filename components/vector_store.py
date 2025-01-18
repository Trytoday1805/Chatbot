from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_vectorstore(chunks):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        return FAISS.from_texts(texts=chunks, embedding=embeddings)
    except Exception as e:
        return f"Lỗi khi tạo vector store: {str(e)}"
