import os
import fitz  # PyMuPDF for reading PDFs


# Function to convert PDF to text and save as .txt files
def convert_pdfs_to_text(pdf_folder, text_folder):
    # Create the folder for text files if it doesn't exist
    if not os.path.exists(text_folder):
        os.makedirs(text_folder)

    # Loop through all the PDF files in the folder
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, file_name)
            text_file_name = os.path.splitext(file_name)[0] + ".txt"
            text_file_path = os.path.join(text_folder, text_file_name)

            # Open PDF and extract text
            with fitz.open(file_path) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()

            # Save the extracted text to a .txt file
            with open(text_file_path, "w", encoding="utf-8") as text_file:
                text_file.write(text)
            print(f"Converted {file_name} to {text_file_name}")


if __name__ == "__main__":
    # Specify the folder with your PDFs
    pdf_folder = "Data"
    # Specify the folder where you want to save .txt files
    text_folder = "DataTxt"
    convert_pdfs_to_text(pdf_folder, text_folder)
