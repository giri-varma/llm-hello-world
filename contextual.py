import gpt4all
import pickle
from sentence_transformers import SentenceTransformer


# all-MiniLM-L6-v2 is the fastest mode in the
# size bracket. You can also use other models
# for better quality or choose smaller models
# for performance. 
# https://www.sbert.net/docs/pretrained_models.html#model-overview
transformer = SentenceTransformer('all-MiniLM-L6-v2')
# https://gpt4all.io/index.html
gpt = gpt4all.GPT4All('all-MiniLM-L6-v2-f16.gguf', n_threads=8)

# constants
corpus_file = 'corpus.pkl'
idx_file = 'index.pkl'

# Load the corpus
print(f"Loading corpus from {corpus_file}...")
with open(corpus_file, 'rb') as f:
    corpus = pickle.load(f)

# Load the index
print(f"Loading index from {idx_file}...")
with open(idx_file, 'rb') as f:
    idx = pickle.load(f)

# Get user question and answer in loop
# User query
# TODO: Get the query at runtime
query = "What are labels?"

# Embed the query
xq = transformer.encode([query])
k = 1
D, I = idx.search(xq, k)

# Nearest neighbour from corpus as per FAISS
text = corpus[I[0][0]]
print(len(text))
messages = [
    {
        "role": "user", 
        "content": (
            "Take the context from context xml tags and answer the question "
            f" between question xml tags. <context>{text}<context>"
            f"<question>{query}<question>"
        )
    }
]
prompt = gpt._format_chat_prompt_template(messages)
# TODO: complains that the prompt is too big,
# needs some prompt enigneering to make this short
response = gpt.generate(prompt=prompt)
print(response)