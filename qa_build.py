from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Constants
DATA_PATH = './data' # Root data folder path; ideally would be an index of document store
IDX_PATH = './index' # Location of the vector store index; ideally a persistent vector store
CHUNK_SIZE = 500 # Size of data chunks that are also semantically meaningful for better context
CHUNK_OVERLAP = 10 # Maximum overlap between subsequent chunks to keep maintain semanticn meaning
TRANSFORMER_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

print(f"Loading the docs in {DATA_PATH}...")
loader = PyPDFDirectoryLoader(DATA_PATH)
documents = loader.load()

print("Splitting the data into smaller yet semantically meanigful chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
splits = text_splitter.split_documents(documents)
print(f"Total number of data chunks: {len(splits)}")

print("Embedding the data chunks...")
embeddings = HuggingFaceEmbeddings(model_name=TRANSFORMER_MODEL, model_kwargs={'device': 'cpu'})

print(f"Indexing the embeddings and persisting them in {IDX_PATH}...")
db = FAISS.from_documents(splits, embeddings)
db.save_local(IDX_PATH)

print("Done")