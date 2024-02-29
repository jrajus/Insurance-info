import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import google.generativeai as genai
from pathlib import Path
import warnings
from io import BytesIO
import tempfile
import json
import pandas as pd
import os
st.set_page_config(layout="wide")
warnings.simplefilter(action='ignore', category=FutureWarning)

@st.experimental_memo
def process_pdf_and_get_info(pdf_name, question_prompt):
    loader = PyPDFLoader(pdf_name)
    data = loader.load_and_split()
    os.environ['GOOGLE_API_KEY'] = user_api_key
    genai.configure(api_key = os.environ['GOOGLE_API_KEY'])
    # Set up context
    context = "\n".join(str(p.page_content) for p in data)

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    texts = text_splitter.split_text(context)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever()

    context_words = "policy details, certificate number, insurance policy, policy type, policy no, policy holder, nominee, agent, address, email, financier, insurer, license, manufacture, contact, information, details, liability, policy, clauses"
    docs = vector_index.get_relevant_documents(context_words)

    prompt_template = """
      Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
      provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
      Context:\n {context}?\n
      Question: \n{question}\n

      Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    response = chain({"input_documents": docs, "question": question_prompt}, return_only_outputs=True)

    info = {}
    info["Document Name"] = pdf_name
    info["Extracted Info"] = list(response.values())[0]
    # Release resources associated with the Chroma instance
    del vector_index

    return info

def truncate_chroma():
    Chroma.truncate()


def displayvals(response_vals, titles):

    # Splitting the 'Extracted Info' into sections
    sections = response_vals['Extracted Info'].strip().split('**')[1:]  # Split and remove empty strings

    # Process sections into a list of (title, content) tuples
    processed_sections = []
    for i in range(0, len(sections), 2):
        title = sections[i].strip(':')
        content = sections[i+1].strip()

        processed_sections.append((title, content))
    
    return processed_sections
            
  
if __name__ == "__main__":
    # Known section titles
    section_titles = [
    
    "Policy Holder Details",
    "Vehicle Details",
    "Insured Values"]
    st.sidebar.header('Know more about your Policy')
    st.sidebar.subheader='Know about your policy'

    user_api_key = st.sidebar.text_input(
        label="Your GoogleAI API key## 👇",
        placeholder="Paste your openAI API key, sk-",
        type="password")
    uploaded_file = st.sidebar.file_uploader("Upload your policy here", type="pdf")

    if uploaded_file:
        try:
        
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
                response_vals = process_pdf_and_get_info(tmp_file_path, "extract policy details, policy holders details, vehicle details and insured values. give the output in an array as four items")
                processed_sections=displayvals(response_vals, section_titles)
                expander1 = st.expander(label=processed_sections[0][0])
                with expander1:
                    st.text(processed_sections[0][1])
                    
                expander2 = st.expander(label=processed_sections[1][0])
                with expander2:
                    st.text(processed_sections[1][1])

                expander2 = st.expander(label=processed_sections[2][0])
                with expander2:
                    st.text(processed_sections[2][1])

                if len(processed_sections)>3:
                    expander3 = st.expander(label=processed_sections[3][0])
                    with expander3:
                        st.text(processed_sections[3][1])
        except Exception as e:
            st.write(f"An error occurred: {e}")       

        question_prompt = st.sidebar.text_area("What do you need to know about the policy:", placeholder="what are the clauses covered in the policy")

        if st.sidebar.button("Submit"):
            try:
                if tmp_file_path:
                    response_vals = process_pdf_and_get_info(tmp_file_path, question_prompt)

                        
                    df = pd.DataFrame(response_vals['Extracted Info'].split('\n'), columns=['Details'])
                    st.subheader(question_prompt)
                    st.table(df)
                    

            except Exception as e:
                st.write(f"An error occurred: {e}")
