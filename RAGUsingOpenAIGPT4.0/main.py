import os
import openai
import time

from dotenv import load_dotenv

# LOAD ENVIRONMENT VARIABLES FROM . env FILE
load_dotenv ()
# SET OPENAI KEY AND MODEL
openai . api_key = os . getenv ( " OPENAI_API_KEY " )
client = openai . OpenAI ( api_key = openai . api_key )
model_name = "gpt-4o" # Any model from GPT - series

def uploadPDFsToVectorStore(client, vectorStoreID, directoryPath):
    try:
        file_ids = {}
        # Get all PDF file paths from the directory
        file_paths = [os.path.join(directoryPath, file) for file in os.listdir(directoryPath) if file.endswith(".pdf")]

        # Iterate through each file and upload to vector store
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            
            # Upload the new file
            with open(file_path, "rb") as file:
                uploaded_file = client.beta.vector_stores.files.upload(vector_store_id=vectorStoreID, file=file)
                print(f"Uploaded file: {file_name} with ID: {uploaded_file.id}")
                file_ids[file_name] = uploaded_file.id

        print(f"All files have been successfully uploaded to vector store with ID: {vectorStoreID}")
        return file_ids

    except Exception as e:
        print(f"Error uploading files to vector store: {e}")
        return None

def get_or_create_vector_store(client, vectorStoreName):
    try:
        # List all existing vector stores
        vector_stores = client.beta.vector_stores.list()
        
        # Check if the vector store with the given name already exists
        for vector_store in vector_stores.data:
            if vector_store.name == vectorStoreName:
                print(f"Vector Store '{vectorStoreName}' already exists with ID: {vector_store.id}")
                return vector_store
        
        # Create a new vector store if it doesn't exist
        vector_store = client.beta.vector_stores.create(name=vectorStoreName)
        print(f"New vector store '{vectorStoreName}' created with ID: {vector_store.id}")
        
        # Upload PDFs to the newly created vector store (assuming 'Upload' is the directory containing PDFs)
        uploadPDFsToVectorStore(client, vector_store.id, 'Upload')
        return vector_store

    except Exception as e:
        print(f"Error creating or retrieving vector store: {e}")
        return None

vector_store = get_or_create_vector_store(client , "RAG Knowledge Base")

def get_or_create_assistant(client, model_name, assistant_name, vector_store_id):
    assistants = client.beta.assistants.list()
    for assistant in assistants.data:
        if assistant.name == assistant_name:
            print("AI Assistant already exists with ID:" + assistant.id)
            return assistant

    assistant = client.beta.assistants.create(
        model=model_name,
        name=assistant_name,
        description="", #Purpose of Assistant
        instructions="", # Specialize instructions and conversation structure
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
        temperature= 0.7,  # Temperature for sampling
        top_p= 0.9  # Nucleus sampling parameter
    )
    print("New AI Assistant created with ID:" + assistant.id)
    return assistant

assistant = get_or_create_assistant(client, model_name, "MyPersonalBot", vector_store.id)

# CREATE THREAD
thread_conversation = {
    "tool_resources": {
        "file_search": {
            "vector_store_ids": [vector_store.id]
        }
    }
}

message_thread = client.beta.threads.create(**thread_conversation)

# INTERACT WITH ASSISTANT
while True:
    user_input = input("Enter your question (or type 'exit' to quit): ")
    
    if user_input.lower() == 'exit':
        print("Exiting the conversation. Goodbye!")
        break
    
    # Add a message to the thread with user input
    message_conversation = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": user_input
            }
        ]
    }

    message_response = client.beta.threads.messages.create(thread_id=message_thread.id, **message_conversation)

    # Initiate a run
    run = client.beta.threads.runs.create(
        thread_id=message_thread.id,
        assistant_id=assistant.id
    )

    # Start fetching messages in real-time
    response_text = ""
    citations = []
    processed_message_ids = set()

    while True:
        response_messages = client.beta.threads.messages.list(thread_id=message_thread.id)
        new_messages = [msg for msg in response_messages.data if msg.id not in processed_message_ids]
        
        for message in new_messages:
            if message.role == "assistant" and message.content:
                message_content = message.content[0].text
                annotations = message_content.annotations
                for index, annotation in enumerate(annotations):
                    message_content.value = message_content.value.replace(
                            annotation.text, f"[{index}]"
                        )
                    if file_citation := getattr(annotation, "file_citation", None):
                        cited_file = client.files.retrieve(file_citation.file_id)
                        citations.append(f"[{index}] {cited_file.filename}")
                words = message_content.value.split()
                for word in words:
                    print(word, end=' ', flush=True)
                    time.sleep(0.05)  
                processed_message_ids.add(message.id)
        
        if any(msg.role == "assistant" and msg.content for msg in new_messages):
            break
        
        time.sleep(1)  

    print("\n") 


