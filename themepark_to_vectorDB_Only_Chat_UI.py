from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DataFrameLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
import pandas as pd
from dotenv import load_dotenv
import os
import streamlit as st
from langchain.schema import SystemMessage, HumanMessage, AIMessage

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load OpenAI API key and Qdrant settings
api_key = os.getenv("MY_OPENAI_KEY") or "default-fake-key-for-testing"
if api_key == "default-fake-key-for-testing":
    print("Warning: Using default key. Check environment variables.")
else:
    # Set to the OpenAI environment variable expected by the library
    os.environ["OPENAI_API_KEY"] = api_key

qdrant_url = os.getenv("MY_QDRANT_URL")
qdrant_key = os.getenv("MY_QDRANT_KEY")

# Ensure Qdrant variables are set
if not qdrant_url or not qdrant_key:
    raise ValueError("Qdrant environment variables are not set. Please set them in GitHub Secrets or locally.")

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

# Initialize OpenAI model
chat = ChatOpenAI(model='gpt-3.5-turbo')

# Initialize the Qdrant client
qdrant_client = QdrantClient(
    url=qdrant_url, 
    api_key=qdrant_key
)

# Initialize the Qdrant vector store
qdrant = Qdrant(
    client=qdrant_client,
    collection_name="chatbot",  
    embeddings=embeddings
)

# Custom prompt function to include wait times and fun recommendations in the response
def get_response(query: str):
    # Perform similarity search and retrieve relevant documents
    results = qdrant.similarity_search(query, k=5)
    
    # Helper function to format wait times
    def format_wait_time(wait_time):
        try:
            # Convert wait time to an integer and handle hours and minutes
            wait_time = int(wait_time)
            hours, minutes = divmod(wait_time, 60)
            if hours > 0:
                return f"{hours} hours and {minutes} minutes" if minutes else f"{hours} hours"
            else:
                return f"{minutes} minutes"
        except (ValueError, TypeError):
            return "Not available"

    # Format the source knowledge to include attraction names and wait times
    source_knowledge = "\n".join([
        f"Attraction: {x.page_content}, Wait Time: {format_wait_time(x.metadata.get('wait_time'))}, Last Live Update: {(x.metadata.get('last_updated'))} PT"
        for x in results
    ])
    
    # Construct the augmented prompt with the extracted knowledge
    augment_prompt = f"""Using the contexts below, answer the query in two parts:

    First, answer the question about wait times, add the Last Live Update using the PT, Then answer the Human any other question. White a cool sentence

    Contexts:
    {source_knowledge}

    Query: {query}"""

    # Initialize the conversation messages
    messages = [
        SystemMessage(content="You are a knowledgeable assistant that provides real-time information about attractions, wait times, and special events at various theme parks."),
        HumanMessage(content="Hi AI, can you give me live updates on attractions?"),
        AIMessage(content="Certainly! I can provide you with current wait times and updates on any attraction. Just let me know which one you're interested in.")
    ]
    
    # Add the custom prompt to the conversation messages
    messages.append(HumanMessage(content=augment_prompt))
    
    # Invoke the chat model with the constructed message chain
    response = chat.invoke(messages)
    
    # Return the AI's response content
    return response.content

# Display the test
def deepeval_test(user_query,final_response):
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

    answer_relevancy = AnswerRelevancyMetric()
    faithfulness = FaithfulnessMetric()

    test_case = LLMTestCase(
        input=user_query,
        actual_output=final_response,
        expected_output="Ride is X hours and X minutes, and the wait time for X is X hour and X minutes.",
        retrieval_context=[
            """X has an immersive dinosaur experience, X is a high-speed thrill"""
        ]
    )
    
    answer_relevancy.measure(test_case)
    #print("Score: ", answer_relevancy.score)
    #print("Reason: ", answer_relevancy.reason)

    faithfulness.measure(test_case)
    #print("Score: ", faithfulness.score)
    #print("Reason: ", faithfulness.reason)
    
    result_text = (
        f"**DeepEval Test Results**\n\n"
        f"**Answer Relevancy**\n"
        f"Score: {answer_relevancy.score}\n"
        f"Reason: {answer_relevancy.reason}\n\n"
        f"**Faithfulness**\n"
        f"Score: {faithfulness.score}\n"
        f"Reason: {faithfulness.reason}"
    )
    return result_text


st.set_page_config(page_title="Chat with Disney", page_icon="🤖")
st.title("Chat with Disney")

st.image("chat.png", use_column_width=True)
st.markdown("""
<p>Welcome to the Disney Chat! Explore Disney and global theme park attractions wait times <i> Note: All parks and attractions <a href="https://docs.google.com/spreadsheets/d/1TSIFqfeS6_dQHWvAkgazUOmQ0e72dAeE/edit?usp=sharing&ouid=113797643410883981985&rtpof=true&sd=true" target="_blank">List</a>. Average response time is 10 seconds. DeepEval Scores for Relevancy score is enabled.</i></p>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.info("You can try typing: 'What is the wait time for Xcelerator The Ride and Jurassic World Adventure? Which one is more fun?' ... or 'What's the wait time for Orion? Is it cool?'... or 'Between Raging Spirits and MaXair, which one has a shorter wait? which one is more intense?'")
    
# Capture and process user input
user_query = st.chat_input("Type your message here...")

if user_query:
    # Append user query to chat history
    st.session_state.chat_history.append(HumanMessage(content=user_query))


    # Show a loading spinner while generating the response
    with st.spinner("Working on it..."):
        # Generate and append AI response
        final_response = get_response(user_query)
        st.session_state.chat_history.append(AIMessage(content=final_response))
        
        # Here to execute the test
        deepeval_result = str(deepeval_test(user_query, final_response))
        st.session_state.chat_history.append(AIMessage(content=deepeval_result))

# Display the conversation history using st.chat_message for better visuals
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)


# In[5]:




