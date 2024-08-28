
# run test and test1.py , app.py not updated


import re
from PIL import Image
import pytesseract
from pytesseract import Output
import pandas as pd
import fitz  # PyMuPDF
import cv2
import numpy as np
from paddleocr import PaddleOCR , draw_ocr
import csv

# Enhanced preprocessing function
import cv2
import numpy as np

import cv2

# Enhanced preprocessing function
def preprocess_image(image_path):
    """
    Preprocess the image to improve OCR accuracy.
    Args:
    - image_path (str): Path to the image to preprocess.

    Returns:
    - processed_image_path (str): Path to the preprocessed image.
    """

    # Load the image using OpenCV
    image = cv2.imread(image_path)

   
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # Apply bilateral filter to reduce noise while keeping edges sharp
    # filtered_image = cv2.bilateralFilter(gray_image, 5, 50, 50)

    # # # Apply adaptive thresholding for better binarization
    # processed_image = cv2.adaptiveThreshold(
    #     filtered_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 8
    # )

    # # Optionally, apply dilation to emphasize the text
    # kernel = np.ones((1, 1), np.uint8)
    # processed_image = cv2.dilate(processed_image, kernel, iterations=1)

    # Save the processed image
    processed_image_path = "processed_image.png"
    cv2.imwrite(processed_image_path, gray_image)
    
    return processed_image_path


# OCR and data extraction functions
def apply_ocr_to_image(image):
    """Apply OCR to extract text from the image."""
    # Custom configuration to specify OCR language and whitelist characters
    custom_config = r'--oem 3 --psm 6'
    ocr_results = pytesseract.image_to_data(image, output_type=Output.DICT, config=custom_config)
    return ocr_results

def extract_text(ocr_results):
    """Extract text from OCR results."""
    text = " ".join([ocr_results['text'][i] for i in range(len(ocr_results['text'])) if ocr_results['text'][i].strip()])
    return text

import re

# def extract_invoice_data_from_text(text):
#     """Extract invoice data from text using regular expressions."""
#     print("Extracted Text: \n", text)  # Debugging: Print the extracted text
    
#     invoice_data = {
#         "Invoice Number": "",
#         "Customer Name": "",
#         "Ship To": "",
#         "Date": "",
#         "Ship Mode": "",
#         "Balance Due": "",
#         "Item": "",
#         "Quantity": "",
#         "Amount": ""
#     }
    
#     # Regex for extracting Invoice Number
#     invoice_number_match = re.search(r"INVOICE\s*#\s*(\d+)", text, re.IGNORECASE)
#     if invoice_number_match:
#         invoice_data["Invoice Number"] = invoice_number_match.group(1)

#     # Regex for extracting Customer Name (more tolerant of OCR errors)
#     customer_name_match = re.search(r"‘([^‘]+)‘",text)
#     if customer_name_match:
#         invoice_data["Customer Name"] = customer_name_match.group(1).strip()

#     # Regex for extracting Ship To
#     ship_to_match = re.search(r"(\d{5},\s+[\w\s,'\"`]+)", text)
#     if ship_to_match:
#         invoice_data["Ship To"] = ship_to_match.group(1).strip()

#     # Regex for extracting Date
#     date_match = re.search(r"(\b\w{3}\s+\d{2}[.,]\d{4}\b)", text)
#     if date_match:
#         invoice_data["Date"] = date_match.group(1).replace(",", ".")

#     # Regex for extracting Ship Mode
#     ship_mode_match = re.search(r"Ship Mode:\s*([\w\s]+)", text)
#     if ship_mode_match:
#         invoice_data["Ship Mode"] = ship_mode_match.group(1).strip()

#     # Regex for extracting Balance Due
#     balance_due_match = re.search(r"Balance Due[:\s]+\$([0-9,]+\.\d{2})", text, re.IGNORECASE)
#     if balance_due_match:
#         invoice_data["Balance Due"] = balance_due_match.group(1)

#     # Regex for extracting Item (adjusted for OCR noise)
#     item_match = re.search(r"(\w+\s+Stacking\s+\w+)", text)
#     if item_match:
#         invoice_data["Item"] = item_match.group(1).strip()

#     # Regex for extracting Quantity
#     quantity_match = re.search(r"\s(\d+)\s+size\s+\$\d+\.\d{2}", text)
#     if quantity_match:
#         invoice_data["Quantity"] = quantity_match.group(1)

#     # Regex for extracting Amount
#     amount_match = re.search(r"Total[:\s]+\$([0-9,]+\.\d{2})", text, re.IGNORECASE)
#     if amount_match:
#         invoice_data["Amount"] = amount_match.group(1)

#     # Print extracted fields for debugging
#     for field, value in invoice_data.items():
#         print(f"{field}: {value}")
    
#     return invoice_data



# def save_data_to_excel(data, file_name="invoice_data.xlsx"):
#     """Save the extracted data to an Excel file."""
#     df = pd.DataFrame([data])
#     df.to_excel(file_name, index=False)

# def save_data_to_csv(data, file_name="invoice_data.csv"):
#     """Save the extracted data to a CSV file."""
#     df = pd.DataFrame([data])
#     df.to_csv(file_name, index=False)

# def paddleocr(img_path):
#     result = ocr.ocr(img_path,cls=True)
#     for i in result:
#         print(i)

def save_to_csv(fl, file_name="invoice_data.csv"):
    """
    Save the data from the list to a CSV file with specified titles (without 'Ship Mode').
    
    Args:
    - res (list): The list containing the invoice data.
    - file_name (str): The name of the CSV file to save the data to.
    """
    # Define the column headers
    headers = ["Invoice Number", "Customer Name", "Ship To", "Date", "Balance Due", "Item", "Quantity", "Amount"]

    # Create and write to the CSV file
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(headers)
        
        # Write the data
        writer.writerow(res)
    
    print(f"Data saved to {file_name} successfully.")


def final_list(img):
    from paddleocr import PaddleOCR, draw_ocr
    # pip install spacy
    # python -m spacy download en_core_web_sm


    ocr = PaddleOCR(use_gpu=False, lang='en')
    
    #can use image or pdfs directly

    data = ocr.ocr(img)
    
    
    for page in data:  
        for element in page:  
            text, confidence = element[1] 
            print(f'Text: {text}, Confidence: {confidence}')
    
    text_list = []
    
    
    for page in data:  
        for element in page:  
            text, _ = element[1]  
            text_list.append(text)
            
    
    
    print(text_list)
    
    import spacy
    
    # Load spaCy's pre-trained English NER model
    nlp = spacy.load("en_core_web_sm")
    
    # Function to extract names and locations
    import spacy
    
    # Load spaCy's pre-trained English NER model
    nlp = spacy.load("en_core_web_sm")

# Function to extract names and locations
    def extract_entities(text_list):
        persons = []
        locations = []
        
        # Process each item in the list for names
        for item in text_list:
            doc = nlp(item)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    if ent.text not in persons:
                        persons.append(ent.text)
    
        # Concatenate all text for location extraction
        concatenated_text = " ".join(text_list)
        
        # Process concatenated text for locations
        doc = nlp(concatenated_text)
        for ent in doc.ents:
            if ent.label_ == "GPE" and ent.text not in locations:
                locations.append(ent.text)
        
        return locations

# Get the extracted names and locations
    locations = extract_entities(text_list)
    locations_string = ", ".join(locations)

    print("Extracted Locations:", locations)





    fl = []
    invoice_number = fl.append(text_list[2])
    
    customer_index = text_list.index('Ship Mode:')
    customer_name = text_list[customer_index+2]
    fl.append(text_list[customer_index+2])
    
    ship_index = text_list.index(customer_name)
    ship_name = text_list[ship_index+1]
    fl.append(locations_string )
    
    date_index = text_list.index('Date:')
    date_name = text_list[date_index+1]
    fl.append(text_list[date_index+1])
    
    balance_index = text_list.index('Balance Due:')
    balance_name = text_list[balance_index+1]
    fl.append(text_list[balance_index+1])
    
    item_1_index = text_list.index('Subtotal:')
    item_1_name = text_list[item_1_index-1]
    
    item_2_index = text_list.index('Amount')
    item_2_name = text_list[item_2_index+1]
    
    final_item_name = item_2_name+" "+item_1_name
    fl.append(final_item_name)
    
    qty_index = text_list.index(item_1_name)
    qty_name = text_list[qty_index-3]
    fl.append(text_list[qty_index-3])
    
    total_index = text_list.index('Total:')
    total_name = text_list[total_index+1]
    fl.append(text_list[total_index+1])
    
    
    import spacy

# Load spaCy's pre-trained English NER model
    nlp = spacy.load("en_core_web_sm")


# Function to extract names and locations
    def extract_entities(fl):
        persons = []
    
        
        # Process each item in the list
        for item in fl:
            doc = nlp(item)
            
            # Extract the entities
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    persons.append(ent.text)
                
        
        return persons

# Get the extracted names and locations
    persons= extract_entities(fl)
    print("Extracted Names:", persons)

    fl[1] = persons[0]
    return fl
    
  

 
    

if __name__ == "__main__":

    pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    # ocr = PaddleOCR(use_angle_cls=True,lang='en')
    # Path to your PDF file
    pdf_path = 'invoice_Amy Hunt_36351.pdf'

#     pdf_document = fitz.open(pdf_path)

#     # Iterate through each page and save as image
#     for page_num in range(len(pdf_document)):
#         page = pdf_document.load_page(page_num)
#         pix = page.get_pixmap()
#         output_image_path = f'page_{page_num}.png'
#         pix.save(output_image_path)

#         # Preprocess the image to improve OCR accuracy
#         processed_image_path = preprocess_image(output_image_path)
#         print(processed_image_path)

#         # Open the processed image
#         image = Image.open(proce    # Open the PDF file
# ssed_image_path)
#         image = image.convert("RGB")
        
#         # Resize the image if necessary
#         max_size = 1000
#         if image.width > max_size:
#             new_width = max_size
#             new_height = int(max_size * image.height / image.width)
#             image = image.resize((new_width, new_height), Image.ANTIALIAS)


    res = final_list(pdf_path)
    print(res)  
    save_to_csv(res)
        
        

