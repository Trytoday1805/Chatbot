# Chatbot Import PDF

## Giới thiệu
Ứng dụng ChatBot Import PDF giúp người dùng tương tác với các tài liệu PDF bằng cách truy vấn và nhận câu trả lời từ nội dung trong tệp PDF. Hệ thống sử dụng mô hình ngôn ngữ lớn (LLM) để cung cấp các câu trả lời chính xác, nhanh chóng và dễ dàng.

## Các công nghệ sử dụng
- **Streamlit**: Framework giao diện người dùng để hiển thị ứng dụng và tương tác với người dùng.
- **pdfplumber**: Thư viện để xử lý và trích xuất dữ liệu từ file PDF.
- **HuggingFaceEmbeddings**: Sử dụng mô hình embedding từ Hugging Face để chuyển đổi nội dung PDF thành các vector.
- **FAISS**: Hệ thống tìm kiếm vector để lưu trữ và truy vấn các vector embedding.
- **Langchain**: Quản lý quy trình tạo câu trả lời dựa trên mô hình ngôn ngữ.
- **VinaLLama**: Mô hình ngôn ngữ (LLM) để tạo ra các câu trả lời thông minh.
- **RAG (Retrieval-Augmented Generation)**: Mô hình kết hợp giữa truy vấn dữ liệu và tạo sinh câu trả lời, giúp cung cấp thông tin chính xác hơn bằng cách kết hợp kết quả tìm kiếm và mô hình sinh câu trả lời.
## Cấu Trúc Thư Mục

Dưới đây là cấu trúc thư mục của dự án:

```
Chatbot/
│
├── .idea/                  # Cấu hình của IDE (Pycharm)
├── .venv/                  # Môi trường ảo Python
├── assets/                 # Các tài nguyên như hình ảnh, biểu tượng
├── components/             # Các thành phần chức năng của ứng dụng
│   ├── __pycache__/        # Các tệp biên dịch Python
│   ├── llm.py              # Mã nguồn cho mô hình ngôn ngữ
│   ├── pdf_processing.py   # Mã xử lý tệp PDF
│   ├── test.py             # Tệp kiểm tra
│   ├── vector_store.py     # Lưu trữ vector cho chatbot
│   └── __init__.py         # Tệp khởi tạo cho thư mục components
│
├── app.py                  # Tệp chính để chạy ứng dụng
├── overlapChunk            # Tệp xử lý phân mảnh văn bản
├── requirements.txt        # Các thư viện Python cần thiết
├── testChunkSot            # Tệp kiểm tra xử lý phân mảnh
├── testReadPDF.txt         # Tệp kiểm tra xử lý PDF
├── test_llm.py             # Tệp kiểm tra mô hình ngôn ngữ
└── README.md               # Tệp README cho dự án
```

## Cách Cài Đặt

1. **Cài đặt môi trường ảo (virtual environment)**:
   
   Đầu tiên, bạn cần cài đặt một môi trường ảo để dự án chạy một cách tách biệt với các dự án khác:

   ```bash
   python -m venv .venv
   ```

2. **Kích hoạt môi trường ảo**:
   
   - Trên Windows:
     ```bash
     .\.venv\Scriptsctivate
     ```
   
   - Trên macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

3. **Cài đặt các thư viện cần thiết**:

   Sau khi môi trường ảo đã được kích hoạt, cài đặt các thư viện từ tệp `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

## Cách Chạy Dự Án

Sau khi cài đặt thành công các phụ thuộc, bạn có thể chạy ứng dụng chatbot bằng lệnh sau:

```bash
 streamlit run app.py
```

Ứng dụng sẽ khởi động và bạn có thể bắt đầu tương tác với chatbot.

## Kiểm Tra

Để kiểm tra các thành phần của dự án, bạn có thể sử dụng các tệp kiểm tra đã có sẵn trong thư mục `components/`. Các tệp kiểm tra này bao gồm:

- `test.py` - Kiểm tra các chức năng chính của ứng dụng.
- `test_llm.py` - Kiểm tra mô hình ngôn ngữ.
- `testChunkSot` và `testReadPDF.txt` - Kiểm tra xử lý phân mảnh văn bản và tệp PDF.

Để chạy kiểm tra, bạn có thể sử dụng các công cụ kiểm tra như `pytest`:

```bash
pytest
```

## Công Nghệ Sử Dụng

- **Python 3.x**: Ngôn ngữ chính cho dự án.
- **Generative AI**: Sử dụng mô hình ngôn ngữ để tạo câu trả lời từ các tệp PDF.
- **PDF Processing**: Xử lý các tệp PDF và trích xuất văn bản.
- **Vector Store**: Lưu trữ dữ liệu dưới dạng vector cho chatbot.

## Góp Ý

Nếu bạn có bất kỳ đề xuất hoặc cải tiến nào cho dự án, đừng ngần ngại tạo một **issue** hoặc gửi **pull request**.



Chúc bạn thành công với dự án!
