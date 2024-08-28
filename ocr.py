from paddleocr import PaddleOCR, draw_ocr
# pip install spacy
# python -m spacy download en_core_web_sm


ocr = PaddleOCR(use_gpu=False, lang='en')

#can use image or pdfs directly
img = r'C:\Users\Admin\Desktop\ayman_proj\invoices\invoice_Marc Harrigan_18843 (1).pdf'
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
print(fl)






