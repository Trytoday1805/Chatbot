import torch
import re
import random
from langchain_community.llms import CTransformers
import os


class CustomLLM:
    def __init__(self, model_file):
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file không tồn tại tại đường dẫn: {model_file}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Sử dụng {device}")

        self.model = CTransformers(
            model=model_file,
            model_type="llama",
            max_new_tokens=128,
            temperature=0.2,
            local_files_only=True,
            stop=['<|im_end|>', '\n\n']
        )

        # Thêm danh sách các từ chào hỏi
        self.greetings = {
            'hi', 'hello', 'chào', 'alo', 'hey',
            'xin chào', 'chào bạn', 'hi bạn', 'hello bạn'
        }

        # Các câu trả lời cho chào hỏi
        self.greeting_responses = [
            "Xin chào! Tôi có thể giúp gì cho bạn?",
            "Chào bạn! Bạn cần hỗ trợ gì ạ?",
            "Xin chào, rất vui được gặp bạn! Tôi có thể giúp gì không ạ?",
        ]

    def is_greeting(self, text: str) -> bool:
        # Chuẩn hóa text: lowercase và loại bỏ dấu câu
        normalized_text = re.sub(r'[.,!?]', '', text.lower().strip())
        # Kiểm tra xem có phải là lời chào không
        return normalized_text in self.greetings or any(
            greeting in normalized_text for greeting in self.greetings
        )

    def get_greeting_response(self) -> str:
        # Chọn ngẫu nhiên một câu trả lời chào hỏi
        return random.choice(self.greeting_responses)

    def generate(self, prompt: str) -> str:
        try:
            clean_prompt = self._clean_prompt(prompt)
            response = self.model(clean_prompt)
            clean_response = self._clean_response(response)
            return clean_response
        except Exception as e:
            return f"Lỗi khi sinh văn bản: {str(e)}"

    def _clean_prompt(self, prompt: str) -> str:
        return f"""Hãy trả lời câu hỏi một cách ngắn gọn và chính xác vào trọng tâm câu hỏi dựa trên context được cung cấp.

Context:
{{context}}

Câu hỏi: {prompt}

Trả lời:"""

    def _clean_response(self, response: str) -> str:
        response = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', response)
        response = response.replace('assistant', '').replace('user', '')
        response = ' '.join(response.split())
        return response


def chatbot_response(question: str, vectorstore, llm: CustomLLM) -> str:
    try:
        # Kiểm tra nếu là lời chào
        if llm.is_greeting(question):
            return llm.get_greeting_response()

        if vectorstore is None:
            return "Chưa có dữ liệu để trả lời."

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
        docs = retriever.get_relevant_documents(question)

        if not docs:
            return "Không tìm thấy thông tin liên quan trong tài liệu."

        context = "\n".join([doc.page_content for doc in docs])
        prompt = llm._clean_prompt(question).replace("{context}", context)
        response = llm.generate(prompt)

        return response

    except Exception as e:
        return f"Lỗi khi xử lý câu hỏi: {str(e)}"