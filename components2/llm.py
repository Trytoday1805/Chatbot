from langchain_community.llms import CTransformers
import os


class CustomLLM:
    def __init__(self, model_file):
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file không tồn tại tại đường dẫn: {model_file}")

        self.model = CTransformers(
            model=model_file,
            model_type="llama",
            max_new_tokens=512,  # Giảm độ dài output
            temperature=0.01,  # Giảm temperature để câu trả lời ổn định hơn
            local_files_only=True,
            stop=['<|im_end|>', '\n\n']  # Thêm stop tokens
        )

    def generate(self, prompt: str) -> str:
        try:
            # Làm sạch prompt
            clean_prompt = self._clean_prompt(prompt)
            response = self.model(clean_prompt)
            # Làm sạch response
            clean_response = self._clean_response(response)
            return clean_response
        except Exception as e:
            return f"Lỗi khi sinh văn bản: {str(e)}"

    def _clean_prompt(self, prompt: str) -> str:
        return f"""Hãy trả lời câu hỏi một cách ngắn gọn và chính xác dựa trên context được cung cấp.

Context:
{{context}}

Câu hỏi: {prompt}

Trả lời:"""

    def _clean_response(self, response: str) -> str:
        # Loại bỏ các tag và cleanup response
        response = response.replace('<|im_start|>', '').replace('<|im_end|>', '')
        response = response.replace('assistant', '').replace('user', '')
        # Loại bỏ khoảng trắng thừa
        response = ' '.join(response.split())
        return response