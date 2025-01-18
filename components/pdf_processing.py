import pdfplumber


# Def read file pdf sử dụng thư viện pdfpluber thay cho pdf2
def read_pdf(file_path):
    try:
        # Mở file PDF bằng pdfplumber
        with pdfplumber.open(file_path) as pdf:
            # Trích xuất nội dung từ tất cả các trang
            pdf_text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())

        return pdf_text
    except Exception as e:
        return f"Lỗi khi đọc PDF: {str(e)}"

# Hàm phân tách văn bản thành các câu
def split_into_sentences(text):
    # Tách câu bằng cách sử dụng phương pháp đơn giản (có thể cải tiến thêm nếu cần)
    # Ví dụ này sử dụng ViTokenizer để tách câu ra thành các từ
    sentences = text.split(". ")  # Tách theo dấu chấm và khoảng trắng
    return sentences


# Hàm để chia văn bản thành các chunk hợp lý
def chunk_text(text, chunk_size=500):
    sentences = split_into_sentences(text)

    chunks = []
    current_chunk = ""
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        # Kiểm tra xem câu có vừa với chunk hiện tại không
        if current_length + sentence_length <= chunk_size:
            current_chunk += sentence + " "
            current_length += sentence_length
        else:
            # Nếu câu quá dài, thêm chunk hiện tại và bắt đầu chunk mới
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
            current_length = sentence_length

    # Thêm chunk cuối cùng nếu có
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# # Ví dụ sử dụng
# text = "Đây là một văn bản tiếng Việt. Nó có thể dài hơn một chút. Tất cả các câu đều được chia tách để tạo thành các chunk. Đoạn văn này sẽ được chia thành nhiều phần nhỏ hơn."
# chunks = chunk_text(text)
# print(f"Đã chia thành {len(chunks)} chunk.")
# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i + 1}: {chunk}")
