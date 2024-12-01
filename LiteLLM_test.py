### Get config
from litellm import completion
import configparser
import os

config = configparser.ConfigParser()
config.read("config.ini")


test_message = [{ "content": "Need detail list of popular language","role": "user"}]

### RUN OLAMA

response = completion(
    model="ollama/llama3.2:1b",
    messages=test_message,
    api_base="http://localhost:11434"
)
print(response['choices'][0]['message']['content'])

### RUN COMHARE

cohere_key = config.get("Credentials", "COHARE_TRIAL_KEY")
print(cohere_key)
os.environ["COHERE_API_KEY"] = cohere_key
response = completion(
    model="command-r",
    messages = test_message
)

print(response['choices'][0]['message']['content'])

### RUN DEEPSEEK
deepseek_key = config.get("Credentials", "DEEPSEEK_API_KEY")
print(deepseek_key)
os.environ["DEEPSEEK_API_KEY"] = deepseek_key
response = completion(
    model="deepseek/deepseek-chat",
    messages = test_message
)

print(response['choices'][0]['message']['content'])


### RUN LM STUDIO
os.environ['LM_STUDIO_API_BASE'] = ""
print('LM_STUDIO_API_BASE')
os.environ["LM_STUDIO_API_KEY"]
response = completion(
    model="lm_studio/llama-3.2-3b-instruct",
    messages=test_message,
)
print(response['choices'][0]['message']['content'])


# os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
# os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")

# response = completion(
#     model="gpt-4o-mini",
#     messages=[{"content": "Hello!", "role": "user"}]
# )

# response = completion(
#     model="claude-3-opus-20240229",
#     messages=[{"content": "Hello!", "role": "user"}]
# )


##

