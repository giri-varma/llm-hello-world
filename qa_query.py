from typing import Any
from langchain.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import LLMResult
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from anyio.from_thread import start_blocking_portal #For model callback streaming


langchain.debug=True 

# Constants
IDX_PATH = './index' # vector db path
TRANSFORMER_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
HF_INFERENCE_SVR = "http://localhost:8080/" # Llama2 TGI models host port

print(f"Loading the data and index from ${IDX_PATH}...")
embeddings = HuggingFaceEmbeddings(model_name=TRANSFORMER_MODEL, model_kwargs={'device': 'cpu'})
db = FAISS.load_local(IDX_PATH, embeddings)

llm = HuggingFaceTextGenInference(
    inference_server_url=HF_INFERENCE_SVR,
    max_new_tokens=512,
    top_k=10,
    top_p=0.9,
    typical_p=0.95,
    temperature=0.6,
    repetition_penalty=1,
    do_sample=True,
    streaming=True
)

# Setup the message templates
system_message = {"role": "system", "content": "You are a helpful assistant."}
template = """
[INST]Use the following pieces of context to answer the question. If no context provided, answer like a AI assistant.
{context}
Question: {question} [/INST]
"""

# RAG retriever
retriever = db.as_retriever(search_kwargs={"k": 6})

# Wire up the RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=retriever,     
    chain_type_kwargs={
        "prompt": PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        ),
    }
)

# Finally test it
result = qa_chain({"query": "What are Labels?"})
print(result)