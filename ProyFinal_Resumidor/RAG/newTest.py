import streamlit as st
import os
import together
from typing import Any
from langchain.llms.base import LLM
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma

from imagenes_base64 import pdf_icon, llama_icon, iconouser, iconollama
import base64
from pdf2image import convert_from_path

os.environ["TOGETHER_API_KEY"] = "f8935229473a0d8a3f4709a9ef32533fe365c0cb215ba8c41413b5ca53a5c767"
class TogetherLLM(LLM):
    model: str = "togethercomputer/llama-2-7b-chat"
    together_api_key: str = os.environ["TOGETHER_API_KEY"]
    temperature: float = 0.1
    max_tokens: int = 1024

    class Config:
        extra = 'forbid'
    @property
    def _llm_type(self) -> str:
        return "together"
    def _call(self, prompt: str, **kwargs: Any) -> str:
        if not self.together_api_key:
            raise ValueError("API key is not set.")
        together.api_key = self.together_api_key
        output = together.Complete.create(prompt,
                                          model=self.model,
                                          max_tokens=self.max_tokens,
                                          temperature=self.temperature)
        print(f"modelo: {self.model}")
        if 'choices' in output:
            return output['choices'][0]['text']
        else:
            raise KeyError("The key 'choices' is not in the response.")
        
  # Funcion para traduccion a ingles
#from googletrans import Translator
#import googletrans
#translator = Translator()

def dividir_texto(texto, max_caracteres):
    textos_divididos = []
    texto_actual = ''
    caracteres_actuales = 0

    oraciones = texto.split('.')

    for oracion in oraciones:
        caracteres_actuales += len(oracion) + 1

        if caracteres_actuales <= max_caracteres:
            texto_actual += oracion + '.'
        else:
            textos_divididos.append(texto_actual.strip())
            texto_actual = oracion + '.'
            caracteres_actuales = len(oracion) + 1

    textos_divididos.append(texto_actual.strip())

    return textos_divididos

########################### CREAR CHUNKS #########################################
def chunk_data(docs, chunk_size=800, overlap=100):
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
        )
    return text_spliter.split_documents(docs)
#################################################################################

@st.cache_resource 
def create_embeddings(uploaded_files):
    try:
        chunks = read_documents_chunks(uploaded_files)
        
        if not chunks:
            st.error("No se pudieron crear chunks de los documentos.")
            return None

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        persist_directory = 'db'
        vectordb = Chroma.from_documents(documents=chunks, 
                                         embedding=embeddings,
                                         persist_directory=persist_directory)
        
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        return retriever
    except Exception as e:
        st.error(f"Error al crear embeddings: {str(e)}")
        return None


def get_prompt(instruction, new_system_prompt ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

sys_prompt = """You are a helpful, respectful, and honest assistant. Always provide the most helpful and accurate answer using only the contextual text provided. Do not add any information that is not in the context.

Guidelines:

1. **Relevance**: Only answer questions based on the provided context. Do not use outside knowledge.
2. **Honesty**: If a question does not make sense or is factually incoherent, explain why instead of providing incorrect information.
3. **Integrity**: If you don't know the answer based on the context, do not share false information.
4. **Scope**: If the question is outside the context, inform the user politely that it cannot be answered accurately.
5. **Clarity**: Ensure your answers are clear and concise, avoiding ambiguity or vagueness.
6. **Finality**: Answer the question directly and do not include any additional text after the answer.

Example of handling an irrelevant question:
- "I'm sorry, but this question is outside the provided context and I cannot answer it accurately."

Example of handling a nonsensical question:
- "The question seems factually incoherent, so I cannot provide an accurate answer."

Ensure that your answers are clear and concise, avoiding ambiguity or vague responses."""

instruction = """CONTEXT:/n/n {context}/n

Question: {question}"""

prompt_template = get_prompt(instruction, sys_prompt)
llama_prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": llama_prompt}

def modelo_llm(modelo):
    return TogetherLLM(
        model= modelo,
    )

# Funcion para el historial de chat
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        #st.session_state['generated'] = ["Hola 👋"]
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        #st.session_state['past'] = ["Hola 👋"]
        st.session_state['past'] = []

def conversation_chat(query,chain,history):
    #result =chain({"question":query, "chat_history": history})
    #user_question_eng = traducir(query)
    user_question_eng = query
    llm_response = chain.invoke(user_question_eng)
    for source in llm_response["source_documents"]:
        # Acceder a la metadata para obtener la página
        page_number = source.metadata['page']
        print(f"Fuente: {source.metadata['source']}, Pagina: {page_number}")
    #llm_response = traducir_ingles(llm_response['result'], "spanish")
    llm_response = llm_response['result']
    history.append((query,llm_response))
    return llm_response

# Simulacion de escritura de cada respuesta
import time
def animate_typing(text):
    for char in text:
        st.write(char, end='', flush=True)
        time.sleep(0.05)

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my-form', clear_on_submit=True):
            user_question = st.text_input("",placeholder="Haz una pregunta a tu PDF",key='my-input')
            submit_button = st.form_submit_button(label='Enivar')

        if user_question and submit_button:
            with st.spinner('Generando repuesta...'):
                output = conversation_chat(user_question, chain, st.session_state['history'])
            
            st.session_state['past'].append(user_question)
            st.session_state['generated'].append(output)
    
    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                
                response_user = st.session_state["past"][i]
                st.markdown("""
                            <div class="response">
                                <div class="response-input">
                                    <img src='data:image/png;base64,{iconouser}' alt='Icono' width='40em' height='40em' style='vertical-align: middle;'> 
                                </div>
                                <div class="message user">{respuesta}</div>
                            <div>""".format(iconouser=iconouser,respuesta=response_user), unsafe_allow_html=True)
                
                respuesta = st.session_state["generated"][i]

                st.markdown(
                    """
                        <div class="response">
                            <div class="response-input">
                                <img src='data:image/png;base64,{iconollama}' alt='Icono' width='40em' height='40em' style='vertical-align: middle;'> 
                            </div>
                            <div class="message model">
                                <span class="typewriter">{respuesta}</span>
                            </div>
                        </div>""".format(iconollama=iconollama,respuesta=respuesta),unsafe_allow_html=True)
                
                
                #st.markdown(f'<div class="response"><div class="response-input"></div><div class="message model">{st.session_state['generated'][i]}</div></div>', unsafe_allow_html=True)

st.set_page_config('preguntaDOC')

with open("design.css") as source_des:
    st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)

st.sidebar.markdown(
    """
    <div class="app-name">
        <img src='data:image/png;base64,{pdf_icon}' alt='Icono' width='40em' height='40em' style='vertical-align: middle;'> 
        <h1 style='color: #FFFFFF;' class="app-name-title">PDF Chatify</h1>
    </div>
    """.format(pdf_icon=pdf_icon),
    unsafe_allow_html=True
)


############################### PARA CARGAR DOCUMENTOS #############################
from langchain.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
def load_documents(directory):
    documentPDF = PyPDFDirectoryLoader(directory)
    documents = documentPDF.load()
    return documents
###################################################################################

######################## CARGAR MULTIPLES DOCUMENTOS ############################
#pdf_obj = st.sidebar.file_uploader("Carga tu documento", type="pdf", on_change=st.cache_resource.clear)
uploaded_files = st.sidebar.file_uploader("Carga tus documentos", accept_multiple_files=True)
upload_directory = "documents"
os.makedirs(upload_directory, exist_ok=True)
main_column, input_column = st.columns([2, 1])
##################################################################################

######################## LEER MULTIPLES DOCUMENTOS ###############################
def read_documents_chunks(uploaded_files):
    print('Uploaded_files',uploaded_files)
    documents = []
    
    st.markdown(
        """
        <h2>Documentos PDF cargados</h2>
        """, unsafe_allow_html=True
    )
    
    for i, uploaded_file in enumerate(uploaded_files):
        file_path = os.path.join("documents/", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
            
        doc = PyPDFLoader(file_path).load()
        documents.extend(doc)
        
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        st.markdown(f"**{uploaded_file.name}**")
        st.markdown(
            """
            <iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="600" type="application/pdf"></iframe>
            """.format(base64_pdf=base64_pdf)
        , unsafe_allow_html=True)
    chunks = chunk_data(documents)
    return chunks
##################################################################################


#st.text("Haz una pregunta sobre tu PDF:")
# Si no hay un chat todavia muestra una animacion
model_option = st.sidebar.selectbox(
    "Selecciona el modelo:",
    ["Llama 7 B", "Llama 13 B", "Llama 70 B"]
)

with main_column:
    if not uploaded_files:
        from imagenes_base64 import llama_icon
        st.markdown("""
                <div class="llama-icon">
                    <img src='data:image/png;base64,{llama_icon}' alt='IconoLlama' style='vertical-align: middle;' class='bounce-image'>
                </div>
        """.format(llama_icon=llama_icon), unsafe_allow_html=True)
    
    if uploaded_files:
        initialize_session_state()
        retriever = create_embeddings(uploaded_files)
        
with input_column:
        
    if uploaded_files:
        if model_option == "Llama 7 B":
            modelo = "togethercomputer/llama-2-7b-chat"
            asd = "7 B"
            st.header(f"Pregunta a LLama {asd} 🦙")
            
        elif model_option == "Llama 13 B":
            modelo = "togethercomputer/llama-2-13b-chat"
            asd = "13 B"
            st.header(f"Pregunta a LLama {asd} 🦙")
        elif model_option == "Llama 70 B":
            modelo = "togethercomputer/llama-2-70b-chat"
            asd = "70 B"
            st.header(f"Pregunta a LLama {asd} 🦙")
        
        llm = modelo_llm(modelo)
        qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                            chain_type="stuff",
                                            retriever=retriever,
                                            return_source_documents=True,
                                            chain_type_kwargs=chain_type_kwargs)
        display_chat_history(qa_chain)

