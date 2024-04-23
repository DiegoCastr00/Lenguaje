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

pdf_obj = st.file_uploader("Carga tu documento", type="pdf", on_change=st.cache_resource.clear)

@st.cache_resource 
def create_embeddings(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

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

if pdf_obj:
    knowledge_base = create_embeddings(pdf_obj)
    retriever = knowledge_base.as_retriever(search_kwargs={"k": 5})
    # create the chain to answer questions
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        chain_type_kwargs=chain_type_kwargs,
                                        return_source_documents=True)
    
    user_question = st.text_input("Haz una pregunta sobre tu PDF:")
    
    if user_question:
        llm_response = qa_chain(user_question)
        st.write(process_llm_response(llm_response))    
