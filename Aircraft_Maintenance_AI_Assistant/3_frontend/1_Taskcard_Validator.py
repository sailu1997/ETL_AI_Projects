import subprocess
import streamlit as st
import pytz
import pandas as pd
import os
import pandas as pd
os.environ['TESSDATA_PREFIX'] =  '/usr/share/tesseract-ocr/4.00/tessdata/'
import streamlit as st
import zipfile
import io
import streamlit.config as config
import uuid
from datetime import datetime, timedelta
import shutil
from streamlit_modal import Modal
from common import (
    deptt,
    win2k,
)

tz = pytz.timezone('Asia/Singapore')
config.set_option('server.maxUploadSize', 1000)

st.set_page_config(page_title="TaskCard Validator", page_icon="😇")
html_temp = """
<div style="background-color:brown;padding:10px">
<h2 style="color:white;text-align:center;">TaskCard Validator </h2>
</div>
"""
st.markdown(html_temp,unsafe_allow_html=True)
st.sidebar.header("Taskcard Validator")

def load_task_list(base_dir):
    list_csv = os.path.join(base_dir, 'list.csv')
    if os.path.exists(list_csv):
        return pd.read_csv(list_csv)
    else:
        return pd.DataFrame(columns=['user_name', 'task_name', 'zipfile_name', 'file_count', 'creation_time', 'status'])

def save_task_list(base_dir, task_list):
    list_csv = os.path.join(base_dir, 'list.csv')
    task_list.to_csv(list_csv, index=False)

def add_task(base_dir, user_name, task_name, zipfile_name,file_count, status='Processing'):
    task_list = load_task_list(base_dir)
    user_name=str(user_name)
    new_task = pd.DataFrame({
        'user_name': [user_name],
        'task_name': [task_name],
        'zipfile_name': [zipfile_name],
        'file_count': [file_count],
        'creation_time': [datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')],
        'status': [status]
    })
    task_list = pd.concat([task_list, new_task], ignore_index=True)
    save_task_list(base_dir, task_list)

def delete_task(base_dir, task_name):
    task_list = load_task_list(base_dir)
    task_list = task_list[task_list['task_name'] != task_name]
    save_task_list(base_dir, task_list)
    task_dir = os.path.join(base_dir, task_name)
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)

def update_task_status(base_dir, task_name, status):
    task_list = load_task_list(base_dir)
    task_list.loc[task_list['task_name'] == task_name, 'status'] = status
    save_task_list(base_dir, task_list)

def count_pdfs_in_directory(directory):
    pdf_count = 0
    for root, _, files in os.walk(directory):
        pdf_count += len([file for file in files if file.endswith('.pdf')])
    return pdf_count

def run_background_task(process_output_dir):
    st.write('Starting processing......')
    subprocess.Popen(['python', 'background.py',process_output_dir])

def read_and_display_csv(csv_file, placeholder):
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, keep_default_na=True, na_values=[''])
        df = df.fillna("NULL")

        with placeholder.container():
            st.write('')
            st.info("""
                **Explanation of Table Values:**
                - **YES:** Indicates that the required item is present.
                - **NO:** Indicates that the required item is not present.
                - **NULL:** Indicates that CARD SIGN-OFF/PAGE SIGN-OFF and DATE COMPLETED sections are not present.
            """)
            col1, col2 = st.columns([2,1])
            with col2:
                st.download_button(
                    label="Download whole csv file",
                    data=open(csv_file, 'rb'),
                    file_name=os.path.basename(csv_file),
                    mime='text/csv'
                )
            with col1:
                if st.button('Back'):
                    st.session_state.selected_task = None
                    st.session_state['show_historic'] = False
                    st.rerun()
                
            st.write(f"Total processed pages: {df.shape[0]}")
            st.table(df)
    else:
        placeholder.error("Empty")
        if st.button('Back'):
            st.session_state.selected_task = None
            st.session_state['show_historic'] = False
            st.rerun()
        
def main(base_dir):
    st.write('')
    st.info("Taskcard validation made easy! Please upload the ZipFile and submit to start processing.")
    st.write('')
    col1, col2 = st.sidebar.columns([1,1])
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    task_list = load_task_list(base_dir)
    task_list['user_name'] = task_list['user_name'].astype(str)

    # check the status of task
    for idx, row in task_list.iterrows():
        task_name = row['task_name']
        task_dir = os.path.join(base_dir, task_name)
        if os.path.exists(task_dir):
            # if there is no pdf, change status to Completed
            pdf_count = count_pdfs_in_directory(task_dir)
            if row['status'] =='Processing':
                if pdf_count == 0:
                    update_task_status(base_dir, task_name, 'Completed')
                    st.rerun()
                else:
                    # if the task has been created for 24 hours and still in Processing，change to Failed
                    start_time = datetime.strptime(row['creation_time'], '%Y-%m-%d %H:%M:%S')
                    if datetime.now() - start_time > timedelta(hours=24):
                        update_task_status(base_dir, task_name, 'Failed')
        else:
            update_task_status(base_dir, task_name, 'NA')

    # Create a modal instance
    modal = Modal("", key="delete_modal")

    # Initialize session state for task to delete
    if 'task_to_delete' not in st.session_state:
        st.session_state.task_to_delete = None
    
    if not task_list.empty:
        col1, col2, col3, col4, col5, col6, col7 = st.columns([2.5, 2, 2, 3.5, 2, 2, 2])
        with col1:
            st.write('**Task Name**')
        with col2:
            st.write('**Zip File**')
        with col3:
            st.write('**File Count**')
        with col4:
            st.write('**Creation Time**')
        with col5:
            st.write('**Status**')
        with col6:
            st.write('')  # Placeholder for Delete button header
        with col7:
            st.write('')  # Placeholder for View button header
        for _, row in task_list.iterrows():
            user_name = row['user_name']
            task_name = row['task_name']
            zipfile_name = row['zipfile_name']
            file_count = row['file_count']
            creation_time = row['creation_time']
            status = row['status']

            col1, col2, col3, col4, col5, col6,col7 = st.columns([2.5, 2, 2, 3.5, 2,2,2])
            with col1:
                st.write(user_name)
            with col2:
                st.write(zipfile_name)
            with col3:
                st.write(file_count)
            with col4:
                st.write(creation_time)
            with col5:
                st.write(status)
            with col6:
                if st.button("Delete", key=f"delete_{task_name}"):
                    st.session_state.task_to_delete = row['task_name']
                    modal.open()
            with col7:
                if st.button("View", key=f"view_{task_name}"):
                    st.session_state.selected_task = task_name
                    st.session_state['show_historic'] = True
                    st.rerun() 
    else:
        st.info('Cannot find any tasks, please upload a new task!')

    with st.sidebar:
        uploaded_zip = st.file_uploader("Upload a Zipfile of PDFs", type="zip")
        user_name = st.sidebar.text_input("Enter a Task Name")
        button = st.button("Submit")

    if modal.is_open():
        with modal.container():
            _,coly = st.columns([1,3])
            with coly:
                st.write('**Are you sure you want to delete this task?**')
            st.write('')
            st.write('')
            _,confirm, cancel = st.columns([1,2,2])
            with confirm:
                if st.button("Confirm"):
                    delete_task(base_dir, st.session_state.task_to_delete)
                    st.session_state.task_to_delete = None
                    modal.close()
                    st.rerun()
            with cancel:
                if st.button("Cancel"):
                    st.session_state.task_to_delete = None
                    modal.close()
    if button and uploaded_zip and user_name:
        if user_name in task_list['user_name'].values:
            st.error(f"The user name '{user_name}' already exists. Please choose a different name.")        
        else:
            try:
                with zipfile.ZipFile(io.BytesIO(uploaded_zip.read()), "r") as z:
                    unique_folder_name = str(uuid.uuid4())
                    process_output_dir  = os.path.join(base_dir, unique_folder_name)
                    if not os.path.exists(process_output_dir ):
                        os.makedirs(process_output_dir )
                    z.extractall(process_output_dir)
                
                pdf_files = []
                for root, _, files in os.walk(process_output_dir):
                    for file in files:
                        if file.endswith('.pdf'):
                            pdf_files.append(os.path.join(root, file))

                file_count = len(pdf_files)
                zipfile_name = uploaded_zip.name
                add_task(base_dir, user_name, unique_folder_name, zipfile_name, file_count, 'Processing')
                run_background_task(process_output_dir)

                st.success(f"File '{uploaded_zip.name}' uploaded successfully as task '{unique_folder_name}'")
                st.rerun()
    
            except zipfile.BadZipFile as e:
                st.warning(f"There was an issue with the ZIP file: {str(e)}")
            except Exception as e:
                st.warning(f"There was an error processing the ZIP file: {str(e)}")
    
def historic_page(base_dir):

    # Display historic data
    selected_task = st.session_state.selected_task
    task_dir = os.path.join(base_dir, selected_task)
    result_csv = os.path.join(task_dir, 'result.csv')
         
    read_and_display_csv(result_csv, st.empty())

    # Back button to go back to the main page

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "w2k_hash" not in st.session_state:
    st.session_state.w2k_hash = ""
if "w2k" not in st.session_state:
    st.session_state.w2k = ""
if "message" not in st.session_state:
    st.session_state.message = ""


if "history7" not in st.session_state:
    st.session_state.history7 = []

if st.session_state.authenticated is True and st.session_state.w2k in win2k and st.session_state.dept in deptt:
    base_dir = f"./b_OutputData/{st.session_state.w2k}"
    if 'show_historic' not in st.session_state:
        st.session_state['show_historic'] = False
    # Show either the main page or the historic data page based on session state
    if st.session_state['show_historic']:
        historic_page(base_dir)
    else:
        main(base_dir)

else: st.warning("Oops you may not have access to this page at the moment!")