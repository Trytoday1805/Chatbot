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
# import torch
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from nltk.corpus import stopwords
# import re
# def create_vectorstore(chunks):
#     try:
#         embeddings = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-mpnet-base-v2",
#             model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
#         )
#         return FAISS.from_texts(texts=chunks, embedding=embeddings)
#     except Exception as e:
#         return f"Lỗi khi tạo vector store: {str(e)}"
# import torch
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import faiss
# from typing import List, Dict, Any
#
#
# def initialize_model(use_gpu: bool = True) -> SentenceTransformer:
#     """Initialize the SentenceTransformer model with proper GPU settings."""
#     if use_gpu and torch.cuda.is_available():
#         device = torch.device('cuda')
#         print(f"Using GPU: {torch.cuda.get_device_name(0)}")
#     else:
#         device = torch.device('cpu')
#         print("Using CPU")
#
#     return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
#
#
# def create_vectorstore(chunks: List[str],
#                        metadata: List[Dict[str, Any]],
#                        batch_size: int = 64,
#                        use_gpu: bool = True) -> Dict:
#     """Create a vector store with GPU support if available."""
#     try:
#         # Initialize model with GPU support
#         model = initialize_model(use_gpu)
#
#         # Process in batches
#         all_embeddings = []
#         all_metadata = []
#
#         for i in range(0, len(chunks), batch_size):
#             batch = chunks[i:i + batch_size]
#             # Get embeddings on GPU if available
#             embeddings_batch = model.encode(
#                 batch,
#                 convert_to_tensor=True,
#                 show_progress_bar=True,
#                 device=model.device
#             )
#             # Move to CPU for numpy conversion
#             all_embeddings.append(embeddings_batch.cpu().numpy())
#             all_metadata.extend(metadata[i:i + batch_size])
#
#         # Combine all embeddings
#         embeddings = np.concatenate(all_embeddings, axis=0)
#
#         # Initialize FAISS index
#         dimension = embeddings.shape[1]
#
#         if use_gpu and torch.cuda.is_available():
#             # Create GPU index
#             res = faiss.StandardGpuResources()
#             config = faiss.GpuIndexFlatConfig()
#             config.device = 0  # GPU device number
#             index = faiss.GpuIndexFlatL2(res, dimension, config)
#         else:
#             # Create CPU index
#             index = faiss.IndexFlatL2(dimension)
#
#         # Add vectors to index
#         index.add(embeddings.astype(np.float32))
#
#         return {
#             'index': index,
#             'metadata': all_metadata
#         }
#
#     except Exception as e:
#         raise Exception(f"Error creating vector store: {str(e)}")
#
#
# def query_vectorstore(vectorstore: Dict,
#                       query: str,
#                       model: SentenceTransformer,
#                       top_k: int = 5) -> List[Dict[str, Any]]:
#     """Query the vector store using GPU if available."""
#     try:
#         index = vectorstore['index']
#         metadata = vectorstore['metadata']
#
#         # Encode query using same device as model
#         query_embedding = model.encode(
#             [query],
#             convert_to_tensor=True,
#             show_progress_bar=False,
#             device=model.device
#         )
#
#         # Move to CPU for FAISS if using CPU index
#         if not isinstance(index, faiss.GpuIndex):
#             query_embedding = query_embedding.cpu()
#
#         # Convert to numpy and ensure float32
#         query_embedding = query_embedding.numpy().astype(np.float32)
#
#         # Search
#         distances, indices = index.search(query_embedding, top_k)
#
#         # Return results
#         return [metadata[idx] for idx in indices[0]]
#
#     except Exception as e:
#         raise Exception(f"Error querying vector store: {str(e)}")
#
#
# # Example usage
# if __name__ == "__main__":
#     # Sample data
#     chunks = [
#         "Đây là câu đầu tiên của văn bản.",
#         "Câu thứ hai có nội dung liên quan.",
#         "Một số thông tin bổ sung cho câu thứ ba.",
#         "Câu thứ tư chứa thông tin quan trọng.",
#         "Câu cuối cùng để hoàn thành ví dụ."
#     ]
#
#     metadata = [
#         {"id": 1, "keyword": "câu 1", "extra_info": "Thông tin về câu 1"},
#         {"id": 2, "keyword": "câu 2", "extra_info": "Thông tin về câu 2"},
#         {"id": 3, "keyword": "câu 3", "extra_info": "Thông tin về câu 3"},
#         {"id": 4, "keyword": "câu 4", "extra_info": "Thông tin về câu 4"},
#         {"id": 5, "keyword": "câu 5", "extra_info": "Thông tin về câu 5"}
#     ]
#
#     try:
#         # Initialize with GPU support
#         use_gpu = torch.cuda.is_available()
#         print(f"GPU available: {use_gpu}")
#
#         if use_gpu:
#             print(f"GPU Device: {torch.cuda.get_device_name(0)}")
#             print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 2:.0f}MB")
#
#         # Create vectorstore
#         vectorstore = create_vectorstore(chunks, metadata, batch_size=2, use_gpu=use_gpu)
#         print("Vectorstore created successfully!")
#
#         # Query example
#         model = initialize_model(use_gpu)
#         query = "Thông tin về câu thứ hai"
#         results = query_vectorstore(vectorstore, query, model, top_k=3)
#         print("Query results:", results)
#
#     except Exception as e:
#         print(f"Error: {str(e)}")
