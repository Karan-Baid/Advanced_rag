import streamlit as st
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import os
import getpass
from pydantic import BaseModel
from typing import List

from dotenv import load_dotenv
load_dotenv()


os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["COHERE_API_KEY"]=os.getenv("COHERE_API_KEY")

embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
)

class QueryVariations(BaseModel):
    queries: List[str]

st.title("RAG app") 
st.write("Upload Pdf's and chat with their content")

api_key=os.getenv("GROQ_API_KEY")

def reciprocal_rank_fusion(chunks_list,k=60,verbose=True):
    rrf_scores=defaultdict(float)
    all_unique_chunks={}

    chunk_id_map={}
    chunk_counter= 1

    for query_idx,chunks in enumerate(chunks_list,1):
        for position,chunk in enumerate(chunks,1):
            chunk_content= chunk.page_content
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content]=f"Chunk_{chunk_counter}"
                chunk_counter+=1
            chunk_id=chunk_id_map[chunk_content]
            all_unique_chunks[chunk_content]=chunk
            position_score=1/(k+position)
            rrf_scores[chunk_id]+=position_score

    sorted_chunks=sorted(
        [(all_unique_chunks[chunk_content],score) for chunk_content,score in rrf_scores.items()],key=lambda x:x[1],
        reverse=True
    )
    return sorted_chunks   

if api_key:
    
    llm=ChatGroq(groq_api_key=api_key,model_name="llama-3.1-8b-instant")

    session_id=st.text_input("Session ID",value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store={}

    uploaded_files=st.file_uploader("Choose A PDf file",type="pdf",accept_multiple_files=True)
    
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)


        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})    

        contextualize_q_system_prompt=("""
            Given a chat history and the latest user question which might reference context in the chat history, 
            formulate a standalone question which can be understood without the chat history. Do NOT answer the question, 
            just reformulate it if needed and otherwise return it as is.
        """)

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
        )
        
        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)


        system_prompt = ("""
            You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. If you don't know the answer, 
            say that you don't know. 
            \n\n
            {context}
        """)

        qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
        )
        
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")
        llm_tool=llm.with_structured_output(QueryVariations)

        prompt = f"""Generate 3 different variations of the following query: {user_input}
        that would help retrieve relevant information from the documents .
        Return 3 alternative that approach the same question from differnet angles.    
        """
        response=llm_tool.invoke(prompt)
        query_variations=response.queries
        all_retrieval_results=[]
        for i,query in enumerate(query_variations,1):
            docs=retriever.invoke(query)
            all_retrieval_results.append(docs)
            
        fused_results=reciprocal_rank_fusion(all_retrieval_results,k=60,verbose=True)
        if user_input:
            session_history=get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":session_id}
                },  
            )
            
            st.write("Assistant:", response['answer'])
            

else:
    st.warning("Please enter the Groq API Key")










