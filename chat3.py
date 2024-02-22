from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# OpenAI key not needed since if we use a local GPT4All server
model = ChatOpenAI(
    openai_api_base='http://localhost:4891/v1',
    openai_api_key='...',
    model_name="orca-mini-3b-gguf2-q4_0.gguf",
    temperature=0
)

# Customize the prompt for a specific domain
prompt_template = ChatPromptTemplate.from_messages([
    ("human", "{question}"),
])

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