import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Function to read all text files and prepare them for vector embedding
def load_text_files(text_folder):
    texts = []
    for file_name in os.listdir(text_folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(text_folder, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                texts.append(file.read())
    return texts

# Create FAISS index from text files
def create_faiss_index(text_folder, index_path, embedding_model='sentence-transformers/all-MiniLM-L6-v2'):
    texts = load_text_files(text_folder)
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vector_store = FAISS.from_texts(texts, embeddings)

    # Save the FAISS index to disk
    vector_store.save_local(index_path)
    print(f"FAISS index saved to {index_path}")

if __name__ == "__main__":
    # The folder where text files are saved
    text_folder = "DataTxt"
    # The path where you want to save the FAISS index
    index_path = "DataIndex"
    create_faiss_index(text_folder, index_path)
