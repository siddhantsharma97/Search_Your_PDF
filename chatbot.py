import streamlit as st
from transformers import AutoModelForSeq2SeqLM , AutoTokenizer
from transformers import pipeline
import torch
import base64
import textwrap
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from constants import CHROMA_SETTINGS

checkpoint="C:\\Users\Siddhant\Documents\\Running-Llama2-on-CPU-Machine\\model\\llama-2-7b-chat.ggmlv3.q4_1.bin"
tokenizer =AutoTokenizer.from_pretrained("C:\\Users\Siddhant\Documents\\Running-Llama2-on-CPU-Machine\\model\\llama-2-7b-chat.ggmlv3.q4_1.bin")
model=AutoModelForSeq2SeqLM.from_pretrained(
    "C:\\Users\\Siddhant\\Documents\\Running-Llama2-on-CPU-Machine\\model\\llama-2-7b-chat.ggmlv3.q4_1.bin",
    device_map="auto",
    offload_folder="offload",
    torch_dtype = torch.float32
)

def llm_pipeline():
    pipe=pipeline(
        'text2text-generation',
        model=model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95
    )

    local_llm=HuggingFacePipeline(pipeline=pipe)
    return local_llm

def qa_llm():
    llm=llm_pipeline()
    embeddings=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db=Chroma(embedding_function=embeddings,persist_directory="db",client_settings=CHROMA_SETTINGS)
    retriever=db.as_retriever()
    qa=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True

    )
    return qa

def process_answer(instruction):
    response=''
    instruction=instruction
    qa=qa_llm()
    generated_text=qa(instruction)
    answer=generated_text["result"]
    return answer, generated_text

def main():
    st.title("Search your pdf")
    with st.expander("About the app"):
        st.markdown("This is a GenAI pdf QA App")
    question=st.text_area("Enter your question")
    if st.button("Search"):
        st.info("Your question: "+question)
        st.info("Your Answer")
        answer,metadata=process_answer(question)
        st.write(answer)
        st.write(metadata)

if __name__ == '__main__':
    main()


