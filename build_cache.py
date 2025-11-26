import os, glob
import fitz
from sentence_transformers import SentenceTransformer, util
import pickle

DATASET_FOLDER = "resume_folder"
CACHE_FILE = "cached_embeddings.pkl"

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text(path):
    text = ""
    with fitz.open(path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

texts = []
files = []

for pdf in glob.glob(os.path.join(DATASET_FOLDER, "*.pdf")):
    t = extract_text(pdf)
    if len(t.strip()) > 20:
        texts.append(t)
        files.append(pdf)

embeddings = model.encode(texts, convert_to_tensor=True)

pickle.dump({"embeddings": embeddings, "files": files}, open(CACHE_FILE, "wb"))

print("Caching completed!")
