# pip3 install transformers
# python3 deepseek_v2_tokenizer.py
import transformers
def deepseek_tokinizer(text):
        chat_tokenizer_dir = "./"
        tokenizer = transformers.AutoTokenizer.from_pretrained(
                chat_tokenizer_dir, trust_remote_code=True
                )
        return tokenizer.encode(text)

result = deepseek_tokinizer("Привет мой старый друг!")
print(result)
print(len(result))
