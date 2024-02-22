import faiss
import os
import pickle
import PyPDF2
from sentence_transformers import SentenceTransformer


# all-MiniLM-L6-v2 is the fastest mode in the
# size bracket. You can also use other models
# for better quality or choose smaller models
# for performance. 
# https://www.sbert.net/docs/pretrained_models.html#model-overview
transformer = SentenceTransformer('all-MiniLM-L6-v2')

# TODO: Get the data from command-line arg
data_dir = 'data/k8s-docs'

# constants
corpus_file = 'corpus.pkl'
idx_file = 'index.pkl'

# Read the corpus
corpus = []
os.listdir(data_dir)
for filename in os.listdir(data_dir):
    print(f"Loading file {filename}...")
    
    with open(f"{data_dir}/{filename}", 'rb') as f:
        pdf = PyPDF2.PdfReader(f)
        text = ''
        for i in range(len(pdf.pages)):
            text += pdf.pages[i].extract_text()
        corpus.append(text)
        
assert len(corpus) == len(os.listdir(data_dir))

# Store the corpus
print(f"Storing corpus of size:{len(corpus)}...")
with open(corpus_file, 'wb') as f:
    pickle.dump(corpus, f)

# Build the embeddings
print('Building embeddings...')
embeddings = transformer.encode(corpus)
assert embeddings.shape[0] == len(corpus)

# Build the index
print('Building the transformed index for the given corpus...')
d = embeddings.shape[1] # Dimension
idx = faiss.IndexFlatL2(d)
idx.add(embeddings)

# Store the index
print(f"Storing the index in {idx_file}...")
with open(idx_file, 'wb') as f:
    pickle.dump(idx, f)

print('Done')