# import streamlit as st
# import os
# from langchain_groq import ChatGroq
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.embeddings import OllamaEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# import time

# from dotenv import load_dotenv
# load_dotenv()

# ## load the Groq API key
# groq_api_key=os.environ['GROQ_API_KEY']

# if "vector" not in st.session_state:
#     st.session_state.embeddings=OllamaEmbeddings()
#     st.session_state.loader=WebBaseLoader("https://docs.smith.langchain.com/")
#     st.session_state.docs=st.session_state.loader.load()

#     st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
#     st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
#     st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

# st.title("ChatGroq Demo")
# llm=ChatGroq(groq_api_key=groq_api_key,
#              model_name="mixtral-8x7b-32768")

# prompt=ChatPromptTemplate.from_template(
# """
# Answer the questions based on the provided context only.
# Please provide the most accurate response based on the question
# <context>
# {context}
# <context>
# Questions:{input}

# """
# )
# document_chain = create_stuff_documents_chain(llm, prompt)
# retriever = st.session_state.vectors.as_retriever()
# retrieval_chain = create_retrieval_chain(retriever, document_chain)

# prompt=st.text_input("Input you prompt here")

# if prompt:
#     start=time.process_time()
#     response=retrieval_chain.invoke({"input":prompt})
#     print("Response time :",time.process_time()-start)
#     st.write(response['answer'])

#     # With a streamlit expander
#     with st.expander("Document Similarity Search"):
#         # Find the relevant chunks
#         for i, doc in enumerate(response["context"]):
#             st.write(doc.page_content)
#             st.write("--------------------------------")
    

# import streamlit as st
# import os
# from langchain_groq import ChatGroq
# from langchain.document_loaders import CSVLoader
# from langchain.embeddings import OllamaEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# import time
# from dotenv import load_dotenv

# load_dotenv()

# ## load the Groq API key
# groq_api_key = os.environ['GROQ_API_KEY']

# if "vector" not in st.session_state:
#     st.session_state.embeddings = OllamaEmbeddings()
#     st.session_state.loader = CSVLoader("med_data.csv", encoding="utf-8")
#     st.session_state.docs = st.session_state.loader.load()

#     st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
#     st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# st.title("ChatGroq Demo")
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# prompt = ChatPromptTemplate.from_template("""
# Answer the questions based on the provided context only. Please provide the most accurate response based on the question
# <context> {context} </context>
# Questions:{input}
# """)

# document_chain = create_stuff_documents_chain(llm, prompt)
# retriever = st.session_state.vectors.as_retriever()
# retrieval_chain = create_retrieval_chain(retriever, document_chain)

# prompt = st.text_input("Input your prompt here")

# if prompt:
#     start = time.process_time()
#     response = retrieval_chain.invoke({"input": prompt})
#     print("Response time :", time.process_time() - start)
#     st.write(response['answer'])

#     # With a streamlit expander
#     with st.expander("Document Similarity Search"):
#         # Find the relevant chunks
#         for i, doc in enumerate(response["context"]):
#             st.write(doc.page_content)
#             st.write("--------------------------------")



# import streamlit as st
# import os
# from langchain_groq import ChatGroq
# from langchain.document_loaders import CSVLoader
# from langchain.embeddings import OllamaEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# import time
# from dotenv import load_dotenv

# load_dotenv()

# ## load the Groq API key
# groq_api_key = os.environ['GROQ_API_KEY']

# if "qa_pairs" not in st.session_state:
#     st.session_state.embeddings = OllamaEmbeddings()
#     st.session_state.loader = CSVLoader("small_med_data.csv", encoding="utf-8")
#     st.session_state.qa_pairs = st.session_state.loader.load()
#     st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.qa_pairs)
#     st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# st.title("ChatGroq Demo")
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# prompt = ChatPromptTemplate.from_template("""
# Use the provided context to answer the question as accurately as possible.

# Question: {input}
# <context>{context}</context>
# """)

# document_chain = create_stuff_documents_chain(llm, prompt)
# retriever = st.session_state.vectors.as_retriever()
# retrieval_chain = create_retrieval_chain(retriever, document_chain)

# prompt = st.text_input("Input your question here")

# if prompt:
#     start = time.process_time()
#     response = retrieval_chain.invoke({"input": prompt})
#     print("Response time :", time.process_time() - start)
#     st.write(response['answer'])

#     with st.expander("Relevant Information"):
#         for i, doc in enumerate(response["context"]):
#             # st.write(f"Question: {doc['question']}")
#             st.write(f"Answer: {doc.page_content}")
#             st.write("--------------------------------")


import streamlit as st
import os
from typing import Optional, Type
from langchain_groq import ChatGroq
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool

class SymptomAnalysisTool(BaseTool):
    name: str = "Symptom_Analysis"
    description: str = "Use this tool when you need to analyze symptoms of a medical condition"
    return_direct: bool = False

    def _run(self, query: str) -> str:
        retriever = st.session_state.vectors.as_retriever()
        docs = retriever.get_relevant_documents(f"symptoms of {query}")
        return "\n".join([doc.page_content for doc in docs])

    def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async")

class RiskFactorTool(BaseTool):
    name: str = "Risk_Factor_Analysis"
    description: str = "Use this tool when you need to identify risk factors for a medical condition"
    return_direct: bool = False

    def _run(self, query: str) -> str:
        retriever = st.session_state.vectors.as_retriever()
        docs = retriever.get_relevant_documents(f"who is at risk for {query}")
        return "\n".join([doc.page_content for doc in docs])

    def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async")

class TreatmentTool(BaseTool):
    name: str = "Treatment_Information"
    description: str = "Use this tool when you need information about treatments for a medical condition"
    return_direct: bool = False

    def _run(self, query: str) -> str:
        retriever = st.session_state.vectors.as_retriever()
        docs = retriever.get_relevant_documents(f"treatment for {query}")
        return "\n".join([doc.page_content for doc in docs])

    def _arun(self, query: str) -> str:
        raise NotImplementedError("This tool does not support async")

# Initialize the base components
if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.loader = CSVLoader("med_data.csv",encoding="utf-8")
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.vectors = FAISS.from_documents(st.session_state.docs, st.session_state.embeddings)

# Initialize the LLM
llm = ChatGroq(
    groq_api_key=os.environ['GROQ_API_KEY'],
    model_name="mixtral-8x7b-32768"
)

# Create the tools list
tools = [
    Tool(
        name="Medical_Knowledge_Base",
        func=RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vectors.as_retriever()
        ).run,
        description="Use this for general medical questions and information"
    ),
    SymptomAnalysisTool(),
    RiskFactorTool(),
    TreatmentTool()
]

# Initialize conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# Streamlit interface
st.title("Medical Information Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("What would you like to know about?")

if user_input:
    try:
        response = agent.run(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(f"You: {message['content']}")
    else:
        st.write(f"Assistant: {message['content']}")
        
    # Add expander for debugging tool usage
    with st.expander("View Tool Usage"):
        st.write("Tools used in generating this response:")
        for tool in tools:
            if tool.name.lower() in message['content'].lower():
                st.write(f"- {tool.name}: {tool.description}")