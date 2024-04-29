import streamlit as st
import together
from typing import Any, Dict
from pydantic import Extra
from langchain.llms.base import LLM
from langchain.utils import get_from_dict_or_env
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.vectorstores import FAISS
import uuid

st.set_page_config('preguntaDOC')
st.header("Pregunta a LLaMA 🦙")

os.environ["TOGETHER_API_KEY"] = "f8935229473a0d8a3f4709a9ef32533fe365c0cb215ba8c41413b5ca53a5c767"

together.api_key = os.environ["TOGETHER_API_KEY"]

class TogetherLLM(LLM):
    """Together large language models."""

    model: str = "togethercomputer/llama-2-70b-chat"
    """model endpoint to use"""

    together_api_key: str = os.environ["TOGETHER_API_KEY"]
    """Together API key"""

    temperature: float = 0.7
    """What sampling temperature to use."""

    max_tokens: int = 512
    """The maximum number of tokens to generate in the completion."""

    class Config:
        extra = Extra.forbid

    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the API key is set."""
        api_key = get_from_dict_or_env(
            values, "together_api_key", "TOGETHER_API_KEY"
        )
        values["together_api_key"] = api_key
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "together"

    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Call to Together endpoint."""
        together.api_key = self.together_api_key
        output = together.Complete.create(prompt,
                                        model=self.model,
                                        max_tokens=self.max_tokens,
                                        temperature=self.temperature,
                                        )
        # print("Output:", output)  # print the entire output
        if 'choices' in output:
            text = output['choices'][0]['text']
            return text
        else:
            raise KeyError("The key 'choices' is not in the response.")
        
with open("design.css") as source_des:
    st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)

from imagenes_base64 import pdf_icon

st.sidebar.markdown(
    """

    <div class="app-name">
        <img src='data:image/png;base64,{pdf_icon}' alt='Icono' width='40em' height='40em' style='vertical-align: middle;'> 
        <h1 style='color: #FFFFFF;' class="app-name-title">PDF Chatify</h1>
    </div>
    """.format(pdf_icon=pdf_icon),
    unsafe_allow_html=True
)

pdf_obj = st.sidebar.file_uploader("Carga tu documento", type="pdf", on_change=st.cache_resource.clear)

model_option = st.sidebar.selectbox(
    "Selecciona el modelo:",
    ["Llama 7B", "Llama 80B","Open AI"]
)

# Funcion para traduccion a ingles
from googletrans import Translator
translator = Translator()

def traducir(texto_original):
    origen = translator.detect(texto_original).lang
    # Si el texto esta en otro idioma que no sea ingles lo traduce
    if origen != "en":
        #print(f'Traduccion de {origen} a en')
        traduccion = translator.translate(texto_original, dest="en", src=origen).text
        return traduccion
    # En caso contrario devuelve el texto original ya que esta en ingles
    else:
        return texto_original
    
# Funcion para traduccion a otro idioma
import googletrans
def traducir_ingles(texto_ingles, idioma_destino):
    for abreviatura, nombre_idioma in googletrans.LANGUAGES.items():
        if nombre_idioma == idioma_destino:
            destino = abreviatura
    traduccion = translator.translate(texto_ingles, dest=destino, src="en").text
    return traduccion

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

@st.cache_resource 
def create_embeddings(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Traducir text a ingles
    max_caracteres = 14000
    textos_divididos = dividir_texto(text, max_caracteres)
    text = ""
    for textSec in textos_divididos:
        text += textSec

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )        
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    print("Knowledge base created")
    return knowledge_base

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Default LLaMA-2 prompt style
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

sys_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided. Your answers should only answer the question once and not have any text after the answer is done.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. 

If the question is not directly related to the provided context, politely inform the user that the question is outside the context scope and cannot be answered accurately.

Ensure that your answers are clear and concise, avoiding ambiguity or vague responses."""

instruction = """CONTEXT:/n/n {context}/n

Question: {question}"""


get_prompt(instruction, sys_prompt)

llm = TogetherLLM(
    model= "togethercomputer/llama-2-7b-chat",
    temperature = 0.1,
    max_tokens = 1024
)

prompt_template = get_prompt(instruction, sys_prompt)

llama_prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": llama_prompt}

import textwrap

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')
    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

def process_llm_response(llm_response):
    return wrap_text_preserve_newlines(llm_response['result'])


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
    user_question_eng = traducir(query)
    llm_response = chain(user_question_eng)
    llm_response = traducir_ingles(llm_response['result'], "spanish")
    history.append((query,llm_response))
    return llm_response

# Simulacion de escritura de cada respuesta
import time
def animate_typing(text):
    for char in text:
        st.write(char, end='', flush=True)
        time.sleep(0.05)

import threading
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
                st.markdown(f'<div class="response"><div class="response-input"></div><div class="message user">{st.session_state['past'][i]}</div><div>', unsafe_allow_html=True)
                st.markdown(f'<div class="response"><div class="response-input"></div><div class="message model"><span class="typewriter">{st.session_state['generated'][i]}</span></div></div>', unsafe_allow_html=True)
                #st.markdown(f'<div class="response"><div class="response-input"></div><div class="message model">{st.session_state['generated'][i]}</div></div>', unsafe_allow_html=True)

#st.text("Haz una pregunta sobre tu PDF:")
# Si no hay un chat todavia muestra una animacion
if not pdf_obj:
    from imagenes_base64 import llama_icon
    st.markdown("""
            <div class="llama-icon">
                <img src='data:image/png;base64,{llama_icon}' alt='IconoLlama' style='vertical-align: middle;' class='bounce-image'>
            </div>
    """.format(llama_icon=llama_icon), unsafe_allow_html=True)

# Clase para historial de conversacion
class ChatHistory:
    def __init__(self):
        self.history = []

    def add_message(self, user_message, llm_response):
        self.history.append({"user": user_message, "llm_response": llm_response})

    def get_history(self):
        return self.history

chat_history = ChatHistory()

if pdf_obj:
    initialize_session_state()
    knowledge_base = create_embeddings(pdf_obj)
    retriever = knowledge_base.as_retriever(search_kwargs={"k": 5})
    # create the chain to answer questions
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        chain_type_kwargs=chain_type_kwargs,
                                        return_source_documents=True)
    
    display_chat_history(qa_chain)
