from transformers import pipeline

class VietnameseLLM:
    def __init__(self, model_name: str = "NlpHUST/gpt2-vietnamese"):
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name
        )

    MAX_PROMPT_LENGTH = 1024  # Ví dụ, giới hạn chiều dài của prompt

    def generate(self, prompt: str) -> str:
        try:
            if len(prompt.strip()) == 0:
                return "Vui lòng cung cấp câu hỏi hoặc yêu cầu hợp lệ."

            prompt = prompt[:self.MAX_PROMPT_LENGTH]  # Cắt prompt nếu quá dài

            print(f"Prompt: {prompt}")  # In ra prompt để kiểm tra
            print(f"__________________________________________________")
            response = self.generator(
                prompt,
                max_new_tokens=300,
                num_return_sequences=1,
                pad_token_id=self.generator.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.3,
                top_k=50,
                top_p=0.95,
                no_repeat_ngram_size=3
            )

            print(f"Response: {response}")  # In ra response để kiểm tra
            print(f"__________________________________________________")
            generated_text = response[0]['generated_text']
            print(f"Generated Text: {generated_text}")  # In ra văn bản sinh ra để kiểm tra
            print(f"__________________________________________________")
            return generated_text[len(prompt):].strip() if generated_text.startswith(prompt) else generated_text.strip()

        except Exception as e:
            return f"Lỗi khi sinh văn bản: {str(e)}"


if __name__ == "__main__":
    llm = VietnameseLLM()
    prompt = "Hãy liệt kê các nguyên nhân phổ biến gây ho kéo dài kèm theo triệu chứng như đờm, đau họng, khàn tiếng, v.v."

    print(llm.generate(prompt))
