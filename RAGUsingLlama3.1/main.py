import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama


# Load FAISS index
def load_faiss_index(index_path, embedding_model):
    # Load the FAISS index using the same embedding model
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store


# Create the RAG system using FAISS and Ollama (Llama 3.1)
def create_rag_system(index_path, embedding_model='sentence-transformers/all-MiniLM-L6-v2', model_name="llama3.1"):
    # Load the FAISS index
    vector_store = load_faiss_index(index_path, embedding_model)

    # Initialize the Ollama model (Llama3.1)
    llm = Ollama(model=model_name)

    # Create a more detailed prompt template
    prompt_template = """
    You are an expert assistant with access to the following context extracted from documents. Your job is to answer the user's question as accurately as possible, using the context below.

    Context:
    {context}

    Given this information, please provide a comprehensive and relevant answer to the following question:
    Question: {question}

    If the context does not contain enough information, clearly state that the information is not available in the context provided.
    If possible, provide a step-by-step explanation and highlight key details.
    """

    # Create a template for formatting the input for the model
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    # Create a RetrievalQA chain that combines the vector store with the model
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain


# Function to run the RAG system with a user question
def get_answer(question, qa_chain):
    answer = qa_chain.run(question)
    return answer


if __name__ == "__main__":
    # Path to the FAISS index directory
    index_path = "DataIndex"

    # Initialize the RAG system
    rag_system = create_rag_system(index_path)

    # Get user input and generate the answer
    while True:
        user_question = input("Ask your question (or type 'exit' to quit): ")
        if user_question.lower() == "exit":
            print("Exiting the RAG system.")
            break
        answer = get_answer(user_question, rag_system)
        print(f"Answer: {answer}")
