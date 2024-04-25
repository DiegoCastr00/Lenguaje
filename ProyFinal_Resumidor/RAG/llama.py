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

st.sidebar.markdown(
    """

    <div class="app-name">
        <img src='data:image/png;base64,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/w+9nudbCe9DJFNUAEQkMiuwVYLDL7I8x9BoLQnuQyRzVABEJLJijsFJje3GhNcaNaWsSNxUAEQkslyRgQkN/bt537NpCY0tkmkqACISmVsiBWCxV3FKAuOKCCoAIhIHY0DcQ7oxYV6jvR73uCLyHhUAEYnMnI3jHM/hz/O+j079iyRIBUBEIvN4lwEumjMBtJysSJJUAEQkDrUxjjXttYvsiRjHE5E2qACISHRGTUwjrSw458c0loh0QMsBi0h0Hk8BcOeSN35gr8Uxloh0TGcARCQOcRSARatauDKGcUSkE1QARCQO0QuA8eN3LrWlMWQRkU5QARCROEQtACuaC/woliQi0ikqACISh+YoG7vzk7d+aPPjCiMiG6YCICJxmBdh2+ZiQX36eCcAAAEDSURBVK/9i5SaCoCIxOHP3d3Q4LY3LrFX4wwjIhumAiAikVmOn3Z7W+PaOLOISOeoAIhIZHMvtlk4t3Z1O3NueeUi6/bZAxHpPhUAEYlF9SqOx5jZ6Q2M2SsKWu5XJBQVABGJxcs/stWv9GI08C1gRYc3du72ag7U5/5FwrHQAUSk8mxxpg/IVfMl8ozB2Z33Vgt8FeOJonPj6xfb70NnFBERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERERkfL1/9+fCEzrx6wwAAAAAElFTkSuQmCC' alt='Icono' width='40em' height='40em' style='vertical-align: middle;'> 
        <h1 style='color: #FFFFFF;' class="app-name-title">PDF Chatify</h1>
    </div>
    """,
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
    print(text)

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

# Clase para historial de conversacion
class ChatHistory:
    def __init__(self):
        self.history = []

    def add_message(self, user_message, llm_response):
        self.history.append({"user": user_message, "llm_response": llm_response})

    def get_history(self):
        return self.history

chat_history = ChatHistory()



#st.text("Haz una pregunta sobre tu PDF:")

if pdf_obj:
    knowledge_base = create_embeddings(pdf_obj)
    retriever = knowledge_base.as_retriever(search_kwargs={"k": 5})
    # create the chain to answer questions
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        chain_type_kwargs=chain_type_kwargs,
                                        return_source_documents=True)
    
    user_question = st.text_input("",placeholder="Haz una pregunta a tu PDF")

    # Traducir a ingles user_question
    print("Texto",user_question)
    print()
    print()
    print()
    print()
    
    #if user_question:
        #llm_response = qa_chain(user_question)

        
        #st.write(process_llm_response(llm_response))

    if user_question:
        user_question = traducir(user_question)
        
        llm_response = qa_chain(user_question)

        # Llm_response se traduce a español
        llm_response = traducir_ingles(llm_response, "spanish")
            
        chat_history.add_message(user_question, llm_response)
            
        for chat in chat_history.get_history():
            st.markdown(f'<div><div class="response-input"></div><div class="message user">{chat["user"]}</div><div>', unsafe_allow_html=True)
            st.markdown(f'<div><div></div><div class="message model">{chat["llm_response"]}</div></div>', unsafe_allow_html=True)
        