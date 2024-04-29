import streamlit as st
import together
from typing import Any
from langchain.llms.base import LLM
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from imagenes_base64 import pdf_icon, llama_icon, iconouser, iconollama

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
from googletrans import Translator
import googletrans
translator = Translator()

def traducir(texto_original):
    try :
        origen = translator.detect(texto_original).lang
        # Si el texto esta en otro idioma que no sea ingles lo traduce
        if origen != "en":
            #print(f'Traduccion de {origen} a en')
            traduccion = translator.translate(texto_original, dest="en", src=origen).text
            return traduccion
        # En caso contrario devuelve el texto original ya que esta en ingles
        else:
            return texto_original
    except:
        return texto_original
    
# Funcion para traduccion a otro idioma
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
    return knowledge_base


def get_prompt(instruction, new_system_prompt ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

sys_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided. Your answers should only answer the question once and not have any text after the answer is done.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. 

If the question is not directly related to the provided context, politely inform the user that the question is outside the context scope and cannot be answered accurately.

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
    user_question_eng = traducir(query)
    llm_response = chain.invoke(user_question_eng)
    llm_response = traducir_ingles(llm_response['result'], "spanish")
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


pdf_obj = st.sidebar.file_uploader("Carga tu documento", type="pdf", on_change=st.cache_resource.clear)


#st.text("Haz una pregunta sobre tu PDF:")
# Si no hay un chat todavia muestra una animacion
if not pdf_obj:
    from imagenes_base64 import llama_icon
    st.markdown("""
            <div class="llama-icon">
                <img src='data:image/png;base64,{llama_icon}' alt='IconoLlama' style='vertical-align: middle;' class='bounce-image'>
            </div>
    """.format(llama_icon=llama_icon), unsafe_allow_html=True)

model_option = st.sidebar.selectbox(
    "Selecciona el modelo:",
    ["Llama 7 B", "Llama 13 B", "Llama 70 B"]
)


if pdf_obj:
    initialize_session_state()
    knowledge_base = create_embeddings(pdf_obj)
    retriever = knowledge_base.as_retriever(search_kwargs={"k": 5})

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
                                        chain_type_kwargs=chain_type_kwargs)
    display_chat_history(qa_chain)

