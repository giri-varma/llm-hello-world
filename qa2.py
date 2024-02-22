from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2048,
    chunk_overlap=200
)
splits = text_splitter.split_documents(data)

embeddings = GPT4AllEmbeddings()
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
str_parser = StrOutputParser()
model = ChatOpenAI(
    openai_api_base='http://localhost:4891/v1',
    openai_api_key='...',
    model_name="orca-mini-3b-gguf2-q4_0.gguf",
    temperature=0
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | str_parser
)

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
# chain.invoke("What are the approaches to Task Decomposition?")