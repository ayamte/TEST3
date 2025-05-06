from PyPDF2 import PdfReader  
from pdf2image import convert_from_path  
import pytesseract  
import os  
  
def extract_text_from_pdf(pdf_path, output_txt_path):  
    """  
    Extracts text from a PDF file, prints it to the terminal, and saves it to a text file.  
    """  
    reader = PdfReader(pdf_path)  
    # Extract text using PyPDF2  
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])  
  
    # If PyPDF2 failed (image-based PDF), apply OCR  
    if not text.strip():  
        print("PyPDF2 failed to extract text. Using OCR...")  
        images = convert_from_path(pdf_path)  # Convert PDF pages to images  
        text = "\n".join([pytesseract.image_to_string(img) for img in images])  # OCR on each image page  
  
    # Print the extracted text to the terminal  
    print("\n--- Extracted Text ---\n")  
    print(text.strip())  
  
    # Save the extracted text to a file  
    with open(output_txt_path, "w", encoding="utf-8") as f:  
        f.write(text.strip())  
      
    print(f"\nText successfully extracted and saved to {output_txt_path}")  
      
    return text.strip()