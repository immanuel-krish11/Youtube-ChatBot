import streamlit as st
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

st.header("Youtube ChatBot")
st.subheader("Enter the video ID and chat with the video!!")

video_id = st.text_input("Enter the video ID") # get the video ID -> url can also be used but ID needs to be extracted

process_result = st.button("Click Here to proceed") # clicks then load transcripts and others


# load the transcript
yt = YouTubeTranscriptApi()
if(process_result):
    with st.spinner("Fetching Transcript and processing..."):
        try:
            # fetch transcript of youtube video
            transcript_list = yt.fetch(video_id= video_id, languages=["en"])
            all_text = " ".join(snippet.text for snippet in transcript_list)

            # splitting the texts
            splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
            chunks = splitter.create_documents([all_text])

            # create embeddings and storing in faiss(Vector DB)
            embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
            vector_store = FAISS.from_documents(chunks, embeddings)


            # Retriever
            retriever  = vector_store.as_retriever(search_type = "similarity", search_kwargs = {'k': 4})
            
            # augmentation
            llm = ChatOpenAI(model = "gpt-4o-mini", temperature= 0.3)

            prompt = PromptTemplate(
                template= """
                    You are a helpful assistant.
                    Answer only from the provided transcript context.
                    If the context is insufficient, just say you don't know.

                    {context}
                    Question: {question}
                """,
                input_variables= ['context', 'question']
            )


            # retrieve all the 4 similar chunks
            def format_doc(retrieved_docs):
                return "\n\n".join(doc.page_content for doc in retrieved_docs)
            
            parallel_chain = RunnableParallel({
                "context": retriever | RunnableLambda(format_doc),
                "question": RunnablePassthrough()
            })
            
            parser = StrOutputParser()
            main_chain = parallel_chain | prompt | llm | parser

            # set some session variabels 
            st.session_state.chain_ready = True
            st.session_state.qa_chain = main_chain
            st.success("Video processed successfully! you can now ask questions.")



        except TranscriptsDisabled:
            st.error("No Transcript available for this video.")
        except Exception as e:
            st.error(f"Error : {str(e)}")


# continue chat
if st.session_state.get("chain_ready", False):
    user_input = st.chat_input("Enter your question")
    st.markdown("**Input**")
    st.write(user_input)
    if user_input:
        with st.spinner("Searching and Generation answer..."):
            try:
                answer = st.session_state.qa_chain.invoke(user_input)
                st.markdown("**Answer:**")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {str(e)}")
