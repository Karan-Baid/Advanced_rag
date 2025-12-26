import streamlit as st
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever
from langchain_cohere import CohereRerank
import os
from collections import defaultdict
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

def reciprocal_rank_fusion(chunks_list,k=60):
    rrf_scores=defaultdict(float)
    all_unique_chunks={}
    chunk_id_map={}
    chunk_counter=1

    for query_idx,chunks in enumerate(chunks_list,1):
        for position,chunk in enumerate(chunks,1):
            chunk_content=chunk.page_content
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content]=f"Chunk_{chunk_counter}"
                chunk_counter+=1
            chunk_id=chunk_id_map[chunk_content]
            all_unique_chunks[chunk_content]=chunk
            position_score=1/(k+position)
            rrf_scores[chunk_id]+=position_score

    id_to_content={chunk_id:content for content,chunk_id in chunk_id_map.items()}
    
    sorted_chunks=sorted(
        [(all_unique_chunks[id_to_content[chunk_id]],score) for chunk_id,score in rrf_scores.items()],
        key=lambda x:x[1],
        reverse=True
    )
    
    return [doc for doc,score in sorted_chunks]   

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
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        bm25_retriever = BM25Retriever.from_documents(splits)
        bm25_retriever.k = 5
        
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.3, 0.7]
        )    

        
        system_prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. 

{context}"""

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]

        user_input = st.text_input("Your question:")
        
        if user_input:
            llm_tool=llm.with_structured_output(QueryVariations)
            prompt = f"""Generate 3 different variations of the following query: {user_input}
            that would help retrieve relevant information from the documents.
            Return 3 alternatives that approach the same question from different angles."""
            
            response=llm_tool.invoke(prompt)
            query_variations=response.queries
            
            all_retrieval_results=[]
            for query in query_variations:
                docs=hybrid_retriever.invoke(query)
                all_retrieval_results.append(docs)
            
            fused_docs=reciprocal_rank_fusion(all_retrieval_results,k=60)
            
            reranker=CohereRerank(model="rerank-english-v3.0",top_n=5)
            reranked_docs=reranker.compress_documents(documents=fused_docs[:20],query=user_input)
            
            context="\n\n".join([doc.page_content for doc in reranked_docs])
            session_history=get_session_history(session_id)
            
            messages=[("system",system_prompt.format(context=context))]
            
            for msg in session_history.messages:
                if msg.type=="human":
                    messages.append(("human",msg.content))
                elif msg.type=="ai":
                    messages.append(("assistant",msg.content))
            
            messages.append(("human",user_input))
            
            answer=llm.invoke(messages)
            
            session_history.add_user_message(user_input)
            session_history.add_ai_message(answer.content)
            
            st.write("Assistant:",answer.content)
            
            with st.expander("View Sources"):
                for i,doc in enumerate(reranked_docs,1):
                    st.write(f"**Source {i}:**")
                    st.write(doc.page_content[:200]+"...")


else:
    st.warning("Please enter the Groq API Key")


