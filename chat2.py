from langchain_community.llms.gpt4all import GPT4All
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# orca-mini is good for testing. You can also use other models for better
#quality or choose smaller models for performance.
# https://gpt4all.io/index.html
model = GPT4All(
    model="orca-mini-3b-gguf2-q4_0.gguf",
    allow_download=True,
    n_threads=8,
    temp=0,
    verbose=False
)

# Customize the prompt for a specific domain
template = """Question: {question}

Answer: """
prompt_template = PromptTemplate.from_template(template)

# Simple string parser
str_parser = StrOutputParser()

# Chain the customizations
chain = prompt_template | model | str_parser

# Start an interatice session
print("Hi there! How can I help you?")
while True:
    # Get user prompt
    prompt = input("> ")

    # Check exit condition
    if prompt == "exit":
        exit(0)

    # Generate response
    for chunk in chain.stream({"question": prompt}):
        print(chunk)