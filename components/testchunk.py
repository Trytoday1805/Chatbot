from pyvi import ViTokenizer
import nltk

# Đọc văn bản
text = """
Bộ Quốc phòng Mỹ khẳng định rằng họ đang theo dõi một quả bóng bay lạ của Trung Quốc trên không phận quốc tế gần Alaska.
Quả bóng bay này đã bay qua các khu vực của Mỹ và Canada và các quan chức Mỹ lo ngại rằng nó có thể là một công cụ gián điệp.
"""

# Tokenize văn bản
tokens = ViTokenizer.tokenize(text)
print("Tokens:", tokens)

# Tokenize thành câu
sentences = nltk.sent_tokenize(text)

# Hàm để chia thành chunks (chia theo câu)
def chunk_text(text, chunk_size=2):
    # Tokenize văn bản thành các câu
    sentences = nltk.sent_tokenize(text)
    # Chia các câu thành các chunk
    chunks = [sentences[i:i+chunk_size] for i in range(0, len(sentences), chunk_size)]
    return chunks

# Chia văn bản thành các chunk (mỗi chunk 2 câu)
chunks = chunk_text(text, chunk_size=2)

# Hiển thị các chunk
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(" ".join(chunk))
    print()
