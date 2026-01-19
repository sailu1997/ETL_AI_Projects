import pandas as pd
import numpy as np
import os
import sys
import re
import base64
from pdf2image import convert_from_path
import pandas as pd
import pytesseract
import cv2
from openai import AzureOpenAI
os.environ['TESSDATA_PREFIX'] ="C:\Program Files\Tesseract-OCR\tessdata"
import streamlit as st
import shutil

CONFIDENCE_VALUE = 90
TEXT_LENGTH = 2
TOTAL_TEXT_LESSTHAN_CONF = 1

def generate_random_df():
    return pd.DataFrame(np.random.rand(3, 4), columns=['A', 'B', 'C', 'D'])

def write_results_to_dataframe(response_content):
    pattern = r'\b(?:YES|NO|NULL|Rate limit error|\d{2}-\d{2}-\d{4}|\d{2}-[A-Za-z]{3}-\d{2}|\d{2}-[A-Za-z]{3}-\d{4})\b'
    matches = re.findall(pattern, response_content)
    response_dict = {}
    response_dict['TASKCARD'] = ''
    if matches:
        response_dict['SIGNATURE'] = matches[0] if len(matches) > 0 else ''
        response_dict['STAMP'] = matches[1] if len(matches) > 1 else ''
        response_dict['DATE COMPLETED'] = matches[2] if len(matches) > 2 else ''
    else:
        response_dict['SIGNATURE'] = 'NULL'
        response_dict['STAMP'] = 'NULL'
        response_dict['DATE COMPLETED'] = 'NULL'

    if response_dict['SIGNATURE'] == 'YES' and response_dict['STAMP'] == 'YES' and (response_dict['DATE COMPLETED'] != 'NULL' and response_dict['DATE COMPLETED'] != 'NO' and response_dict['DATE COMPLETED'] != 'UNSURE'):
        response_dict['RESULT'] = 'PASS'
    elif response_dict['SIGNATURE'] == 'NULL' and  response_dict['STAMP'] == 'NULL' and response_dict['DATE COMPLETED'] == 'NULL':
        response_dict['RESULT'] = 'NULL'
    else:
        response_dict['RESULT'] = 'FAIL'

    #response_dict['GPT-4o_response'] = response_content

    df = pd.DataFrame([response_dict])
    return df

# Function to convert PDF to PNG
def pdf_to_png(pdf_path, output_folder, pdf_name,table_placeholder,dpi=300):
    results = pd.DataFrame()
    image_ext = "png" 

    # Convert the PDF to images with different DPI
    ocr_dpi = 200
    pages_standard_dpi = convert_from_path(pdf_path, dpi=dpi)
    pages_ocr_dpi = convert_from_path(pdf_path, dpi=ocr_dpi)

    for page_number, (page_standard, page_ocr) in enumerate(zip(pages_standard_dpi, pages_ocr_dpi)):
        # Save the OCR DPI image (200 DPI)
        image_name_200 = f"{pdf_name}_page_{page_number}.{image_ext}"
        image_path_200 = os.path.join(output_folder, '200_dpi', image_name_200)
        image_dir_200 = os.path.dirname(image_path_200)
        if not os.path.exists(image_dir_200):
            os.makedirs(image_dir_200)
        page_ocr.save(image_path_200, image_ext.upper())
        image_path_300 = None

        try:
            ocr_result = check_keyword_in_image(image_path_200)
            if (ocr_result['text'].str.contains('tifies').any()) or (ocr_result['text'].str.contains('SAR-145').any()):
                # Save the standard DPI image (300 DPI)
                image_name_300 = f"{pdf_name}_page_{page_number}.{image_ext}"
                image_path_300 = os.path.join(output_folder, '300_dpi', image_name_300)
                image_dir_300 = os.path.dirname(image_path_300)
                if not os.path.exists(image_dir_300):
                    os.makedirs(image_dir_300)
                page_standard.save(image_path_300, image_ext.upper())

                base64_image = encode_image(image_path_300)
                response_content = process_image(base64_image)
                dataframe = write_results_to_dataframe(response_content)
                results = pd.concat([results, dataframe], ignore_index=True)
            else:
                ocr_result = ocr_result[ocr_result['text'].notna()].reset_index(drop=True)
                ocr_result['text'] = ocr_result['text'].apply(lambda x:re.sub('[^a-zA-Z]+', '', str(x))) # remove punctuations etc
                ocr_result['text_len'] = ocr_result['text'].apply(lambda x:len(x))
                if ((ocr_result['conf'] > CONFIDENCE_VALUE) & (ocr_result['text_len'] > TEXT_LENGTH)).sum() < TOTAL_TEXT_LESSTHAN_CONF:
                    # Save the standard DPI image (300 DPI)
                    image_name_300 = f"{pdf_name}_page_{page_number}.{image_ext}"
                    image_path_300 = os.path.join(output_folder, '300_dpi', image_name_300)
                    image_dir_300 = os.path.dirname(image_path_300)
                    if not os.path.exists(image_dir_300):
                        os.makedirs(image_dir_300)
                    page_standard.save(image_path_300, image_ext.upper())

                    base64_image = encode_image(image_path_300)
                    response_content = process_image(base64_image)
                    dataframe = write_results_to_dataframe(response_content)
                    results = pd.concat([results, dataframe], ignore_index=True)
                else:
                    response_content = '''1. SIGNATURE : NULL 2. STAMP : NULL 3. DATE COMPLETED : NULL '''
                    dataframe = write_results_to_dataframe(response_content)
                    dataframe['TASKCARD'] = pdf_name
                    dataframe['PAGENUMBER'] = f"page_{page_number+1}"
                    results = pd.concat([results, dataframe], ignore_index=True)
        except Exception as e:
            # Save the standard DPI image (300 DPI)
            image_name_300 = f"{pdf_name}_page_{page_number}.{image_ext}"
            image_path_300 = os.path.join(output_folder, '300_dpi', image_name_300)
            image_dir_300 = os.path.dirname(image_path_300)
            if not os.path.exists(image_dir_300):
                os.makedirs(image_dir_300)
            page_standard.save(image_path_300, image_ext.upper())

            base64_image = encode_image(image_path_300)
            response_content = process_image(base64_image)
            dataframe = write_results_to_dataframe(response_content)
            dataframe['TASKCARD'] = pdf_name
            dataframe['PAGENUMBER'] = f"page_{page_number+1}"
            results = pd.concat([results, dataframe], ignore_index=True)
        
        # all_results = pd.concat([st.session_state.all_results, dataframe], ignore_index=True)

        # with table_placeholder:
        #     st.dataframe(st.session_state.all_results[['TASKCARD', 'PAGENUMBER', 'SIGNATURE','STAMP', 'DATE COMPLETED', 'RESULT']])
        if image_path_200:
            os.remove(image_path_200)
        if image_path_300:
            os.remove(image_path_300)
    #delete_png_files(output_folder)
    return results

# Function to process PDFs in a folder
def process_pdfs_in_folder(folder_path, output_dir,table_placeholder,dpi):
    pdf_name = os.path.basename(folder_path)
    results = pdf_to_png(folder_path, output_dir, pdf_name, table_placeholder,dpi)
    return results

# Function to delete PNG files in a folder
def delete_png_files(output_dir):
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(root, file)
                os.remove(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to process image and make API call
def process_image(base64_image):
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    )
    try:
        response = client.chat.completions.create(
            model='GPT4O',
            messages=[
                {"role": "system", "content": "You are an Aircraft Taskcard Validator Expert. Your task is to examine the Taskcard images according to the following instructions:"},
                {"role": "user", "content": [
                    {"type": "text", "text": """
                    1. CARD SIGN-OFF or PAGE SIGN-OFF Section:
                        Check for a section labeled "CARD SIGN-OFF" or "PAGE SIGN-OFF".
                            a. Look for any handwritten signatures within this section.
                                - If a clear handwritten signature is present, respond with YES.
                                - If no clear handwritten signature is present, respond with NO.
                            b. Check for a stamp in this section.
                                - If a stamp is present, respond with YES.
                                - If no stamp is present, respond with NO.
                    2. DATE COMPLETED Section:
                        Check for a section labeled "DATE COMPLETED" under the "CARD SIGN-OFF" or "PAGE SIGN-OFF" section.
                            a. If a date is present in the format DD-MM-YYYY, respond with the date in this format.
                            b. If no date is present, respond with NO.
                    3. If CARD SIGN-OFF or PAGE SIGN-OFF and DATE COMPLETED sections are not present:
                        Respond with NULL for all the three SIGNATURE , STAMP and DATE COMPLETED
                    Please ensure responses are formatted as follows:
                        1. SIGNATURE: [YES/NO/NULL]
                        2. STAMP: [YES/NO/NULL]
                        3. DATE COMPLETED: [DD-MM-YYYY/NO/NULL]
                    """},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]}
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.warning(
            "Oops, you've either hit the rate limit of  GPT4 to make maximum hits in one minute or have breached the models maximum context length is 32768 tokens. Please relogin on Newton and try again in 30 seconds. We apologize for the inconvenience and thank you for your patience."
            )
        return '''1. SIGNATURE : Rate limit error 2. STAMP : Rate limit error 3. DATE COMPLETED : Rate limit error '''
    
# Function to check for the keyword "certifies" in an image
def check_keyword_in_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ocr_result = pytesseract.image_to_data(gray_image,output_type='data.frame')
    ocr_result = ocr_result.dropna(subset=['text']).reset_index(drop=True)
    #ocr_result['text'] = ocr_result['text'].str.lower()
    ocr_result['text'] = ocr_result['text'].astype(str).str.lower()
    
    return ocr_result

def clean_folder_except_result(folder_path):
    # Iterate over all the files and directories in the folder
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        # If the item is a file and its name is not result.csv, delete it
        if os.path.isfile(item_path) and item != 'result.csv':
            os.remove(item_path)
        # If the item is a directory, delete it and its contents
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def process_file(process_output_dir):
    if os.path.exists(process_output_dir):
        #get file namelist of the zip folder
        extracted_files = []
        for root, dirs, files in os.walk(process_output_dir):
            for name in files:
                relative_path = os.path.relpath(os.path.join(root, name), process_output_dir)
                relative_path = relative_path.replace(os.sep, '/')
                if '__MACOSX' not in relative_path:
                    extracted_files.append(relative_path)
            
            for name in dirs:
                relative_path = os.path.relpath(os.path.join(root, name), process_output_dir)
                relative_path = relative_path.replace(os.sep, '/') + '/'
                if '__MACOSX' not in relative_path:
                    extracted_files.append(relative_path)

        #print(sorted(extracted_files))

        table_placeholder = os.path.join(process_output_dir,'result.csv')
        pdf_files = [file for file in extracted_files if file.endswith('.pdf')]
        if len(pdf_files) == 0:
            clean_folder_except_result(process_output_dir)
            return
        all_results = pd.DataFrame()
        for uploaded_file in pdf_files:
            uploaded_file = os.path.join(process_output_dir, uploaded_file)
            df = process_pdfs_in_folder(uploaded_file, process_output_dir, table_placeholder, dpi=300)
            # Replace with your actual column names in the desired order
            column_order = ['TASKCARD', 'PAGENUMBER', 'SIGNATURE', 'STAMP', 'DATE COMPLETED', 'RESULT']  
            df = df[column_order] 
            all_results = pd.concat([all_results, df], ignore_index=True)
            all_results.to_csv(table_placeholder, index=False)
            if os.path.exists(uploaded_file):
                os.remove(uploaded_file)
        clean_folder_except_result(process_output_dir)
        all_results.to_csv(table_placeholder, index=False)
        print('*****',table_placeholder)

    else:
        print(f"Error:file {process_output_dir} doesn't exist")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_output_dir = sys.argv[1]
        # print(process_output_dir)
        process_file(process_output_dir)
    else:
        print("Please provide file path")
