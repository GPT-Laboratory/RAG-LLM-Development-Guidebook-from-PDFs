import os
import openai
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if OPENAI_API_KEY is set
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("Error: OPENAI_API_KEY is not set in the environment. Please set it in the .env file.")

# Set OpenAI key and model
openai.api_key = openai_api_key
client = openai.OpenAI(api_key=openai.api_key)
model_name = "gpt-4o"  # Any model from GPT series

def upload_pdfs_to_vector_store(client, vector_store_id, directory_path):
    try:
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Error: Directory '{directory_path}' does not exist.")
        if not os.listdir(directory_path):
            raise ValueError(f"Error: Directory '{directory_path}' is empty. No files to upload.")
        
        file_ids = {}
        # Get all PDF file paths from the directory
        file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(".pdf")]

        # Check if there are any PDFs to upload
        if not file_paths:
            raise ValueError(f"Error: No PDF files found in directory '{directory_path}'.")

        # Iterate through each file and upload to vector store
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            
            # Upload the new file
            with open(file_path, "rb") as file:
                uploaded_file = client.beta.vector_stores.files.upload(vector_store_id=vector_store_id, file=file)
                print(f"Uploaded file: {file_name} with ID: {uploaded_file.id}")
                file_ids[file_name] = uploaded_file.id

        print(f"All files have been successfully uploaded to vector store with ID: {vector_store_id}")
        return file_ids

    except Exception as e:
        print(f"Error uploading files to vector store: {e}")
        return None

def get_or_create_vector_store(client, vector_store_name):
    if not vector_store_name:
        raise ValueError("Error: 'vector_store_name' is not set. Please provide a valid vector store name.")
    
    try:
        # List all existing vector stores
        vector_stores = client.beta.vector_stores.list()
        
        # Check if the vector store with the given name already exists
        for vector_store in vector_stores.data:
            if vector_store.name == vector_store_name:
                print(f"Vector Store '{vector_store_name}' already exists with ID: {vector_store.id}")
                return vector_store
        
        # Create a new vector store if it doesn't exist
        vector_store = client.beta.vector_stores.create(name=vector_store_name)
        print(f"New vector store '{vector_store_name}' created with ID: {vector_store.id}")
        
        # Upload PDFs to the newly created vector store (assuming 'Upload' is the directory containing PDFs)
        upload_pdfs_to_vector_store(client, vector_store.id, 'Upload')
        return vector_store

    except Exception as e:
        print(f"Error creating or retrieving vector store: {e}")
        return None

vector_store_name = ""  # Ensure this is set to a valid name
vector_store = get_or_create_vector_store(client, vector_store_name)

def get_or_create_assistant(client, model_name, assistant_name, vector_store_id):
    if not assistant_name:
        raise ValueError("Error: 'assistant_name' is not set. Please provide a valid assistant name.")
    
    description = ""  # Ensure Purpose of Assistant is set here
    if not description:
        raise ValueError("Error: 'description' is not set. Please provide a description for the assistant.")
    
    instructions = ""  # Ensure Specialized Instructions for Assistant and Conversation Structure is set here
    if not instructions:
        raise ValueError("Error: 'instructions' is not set. Please provide instructions for the assistant.")
    
    try:
        assistants = client.beta.assistants.list()
        for assistant in assistants.data:
            if assistant.name == assistant_name:
                print("AI Assistant already exists with ID:" + assistant.id)
                return assistant

        assistant = client.beta.assistants.create(
            model=model_name,
            name=assistant_name,
            description=description,  
            instructions=instructions,  
            tools=[{"type": "file_search"}],
            tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
            temperature=0.7,  # Temperature for sampling
            top_p=0.9  # Nucleus sampling parameter
        )
        print("New AI Assistant created with ID:" + assistant.id)
        return assistant

    except Exception as e:
        print(f"Error creating or retrieving assistant: {e}")
        return None

assistant_name = ""  # Ensure this is set to a valid name
assistant = get_or_create_assistant(client, model_name, assistant_name, vector_store.id)

# Create thread
thread_conversation = {
    "tool_resources": {
        "file_search": {
            "vector_store_ids": [vector_store.id]
        }
    }
}

message_thread = client.beta.threads.create(**thread_conversation)

# Interact with assistant
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
