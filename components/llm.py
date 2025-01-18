# #
from transformers import pipeline


class VietnameseLLM:
    def __init__(self, model_name: str = "NlpHUST/gpt2-vietnamese"):
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name
        )

    MAX_PROMPT_LENGTH = 1024

    def generate(self, prompt: str) -> str:
        try:
            if len(prompt.strip()) == 0:
                return "Vui lòng cung cấp câu hỏi hoặc yêu cầu hợp lệ."

            response = self.generator(
                prompt,
                max_new_tokens=500,
                num_return_sequences=1,
                pad_token_id=self.generator.tokenizer.eos_token_id,
                do_sample=False,  # Không lấy mẫu
                no_repeat_ngram_size=3
            )

            generated_text = response[0]['generated_text']
            return generated_text[len(prompt):].strip() if generated_text.startswith(prompt) else generated_text.strip()

        except Exception as e:
            return f"Lỗi khi sinh văn bản: {str(e)}"

# Ví dụ gọi hàm
llm = VietnameseLLM()
prompt = "Nguyên nhân ho kéo dài"
print(llm.generate(prompt))


def chatbot_response(user_input: str, vectorstore, llm: VietnameseLLM) -> str:
    try:
        if vectorstore is None:
            return "Chưa có dữ liệu để trả lời."

        # Tìm kiếm tài liệu liên quan đến câu hỏi của người dùng
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        docs = retriever.invoke(user_input)

        if not docs:
            return "Không tìm thấy thông tin liên quan trong tài liệu."

        # Tạo bối cảnh từ các tài liệu tìm được
        context = "\n".join([doc.page_content for doc in docs])

        # Tạo prompt cho mô hình sinh văn bản
        prompt = f"""Dưới đây là nội dung từ tài liệu:
{context}

Câu hỏi: {user_input}
Trả lời ngắn gọn và chính xác:"""

        # Gọi hàm generate để sinh văn bản
        return llm.generate(prompt)

    except Exception as e:
        # Trả về thông báo lỗi nếu có
        return f"Lỗi khi xử lý câu hỏi: {str(e)}"
# #___________________________________________________________________________
