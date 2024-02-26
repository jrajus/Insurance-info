import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.question_answering import load_qa_chain
import os
from langchain.vectorstores import Chroma
import google.generativeai as genai
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tempfile
import json
import pandas as pd

st.sidebar.text='Know about your policy'
uploaded_file = st.sidebar.file_uploader("Upload your policy here", type="pdf")

@st.experimental_memo
def return_json(pdf_name, info):
    # Convert and write JSON object to file
    file_name = pdf_name + ".json"
    with open(file_name, "w") as outfile: 
        json.dump(info, outfile)

@st.experimental_memo
def process_pdf_and_get_info(pdf_name,question_prompt):
    # st.markdown('Extracting PDF information')

    # Load PDF and extract text
    loader = PyPDFLoader(pdf_name)
    data = loader.load_and_split()

    os.environ['GOOGLE_API_KEY'] = "AIzaSyDqWiUZ676y19mw2BlKBVV7_82nAAKkX9E"
    genai.configure(api_key = os.environ['GOOGLE_API_KEY'])
    
    # Split text into chunks
    context = "\n".join(str(p.page_content) for p in data)
     
    # st.markdown('RecursiveCharacterTextSplitter')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    context = "\n\n".join(str(p.page_content) for p in data)
        

    texts = text_splitter.split_text(context)
        

    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever()

    context_words = "policy details, certificate number, insurance policy, policy type, policy no, policy holder, nominee, agent, address, email, financier, insurer, license, manufacture, contact, information, details, liability, policy, clauses"
    docs = vector_index.get_relevant_documents(context_words)
    
    # Define QA prompt template
    prompt_template = """
      Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
      provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
      Context:\n {context}?\n
      Question: \n{question}\n

      Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Define details categories
    info = {}
    info["Document Name"]= pdf_name

    # st.markdown('ChatGoogleGenerativeAI')
    # Load ChatGoogleGenerativeAI model
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    # st.markdown('chain')
    # Load QA chain
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    # st.markdown('response')
    response = chain({"input_documents": docs, "question": question_prompt}, return_only_outputs=True)
    # st.markdown('response RECEIVED')
    
    # print(response.values())
    info["Extracted Info"]= list(response.values())[0]

    # info=''
    # Return 'response' dictionary
    return info



# Function to parse the multiline text into a structured dictionary
def parse_policy_information(output_text):
    parsed_data = {}
    
    lines = output_text.split('\n')
    # print(lines)
    key = None
    for line in lines:
        # print((line.startswith('*') or line.startswith('_')))
        # if (line.startswith('*') or line.startswith('_')) :  # New key-value pair
            if ':' in line:
                # print(line)
                key, value = line.split(':', 1)  # Split on first colon
                key = key.strip()
                value = value.strip()
                if key in parsed_data:  # Handle multiple lines for the same key
                    parsed_data[key] += ' ' + value
                else:
                    parsed_data[key] = value
            else:
                key = line.strip()  # Key with no immediate value
                parsed_data[key] = ''  # Initialize with empty string
        # elif key and line:  # Continuation of the previous key's value
        #     parsed_data[key] += ' ' + line.strip()
    # print(parsed_data)
    return parsed_data





if uploaded_file :
   #use tempfile because CSVLoader only accepts a file_path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    question_prompt = st.sidebar.text_area("What od you need know about the policy:", "Extract all policy information")
    if st.sidebar.button("Submit"):
        try:
            if tmp_file_path:
                # st.success(f'File {uploaded_file.name} is successfully saved!')
                # st.markdown('starting the process')
                response_vals = process_pdf_and_get_info(tmp_file_path,question_prompt)
                # st.markdown('End process')
                # st.write(response_vals)

                # Parse the 'output_text'

            # parsed_information = parse_policy_information(response_vals['Extracted Info'])

            df=pd.DataFrame(response_vals['Extracted Info'].split('\n'), index=None,columns=['Details'])
            
            
            st.subheader('Policy Information')

            st.table(df)

        except Exception as e:
            st.write(f"An error occurred: {e}")


