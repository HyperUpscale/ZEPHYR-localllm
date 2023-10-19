import os
from dotenv import load_dotenv
import chainlit as cl
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

hft = os.getenv("HUGGING_FACE_TOKEN")

repo_id = "HuggingFaceH4/zephyr-7b-alpha"
model = HuggingFaceHub(
    repo_id=repo_id,
    huggingfacehub_api_token=hft,
    model_kwargs={"temperature": 0.8, "max_new_tokens": 200},)

prompt = PromptTemplate(template="{question}", input_variables=["question"])

@cl.langchain_factory(use_async=False)
def factory():
    chain = LLMChain(prompt=prompt, llm=model, verbose=True)
    return chain
