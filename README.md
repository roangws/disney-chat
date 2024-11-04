# Retrieval-Augmented Generation (RAG) Chatbot with DeepEval Evaluation – Disney Chat

Welcome to the Disney Chatbot! Explore live data for Disney and others theme park attractions! 

I created a Retrieval-Augmented Generation (RAG) chatbot to provide human responses based on real-time data from worldwide theme parks API. The RAG chatbot uses Langchain and integrates with a vector database on Qdrant to enable retrieval and generation capabilities, uses the OpenAI GPT-3.5-turbo model, and I evaluate the results using DeepEval Evaluation. 

Started by Connecting to the ThemeParks API to get data on theme parks and their respective attractions’ waiting line times. Each theme park has a unique ID that serves as a key for accessing multiple attraction details. Once the attraction data is retrieved, it is exported to an Excel file for easier visualization for this example. 

On Qdrant vector database, I clear any previous data and insert the new data. As this is an initial test, I’m manually deleting and uploading the new data. In a full implementation, functions would be added to detect updates automatically. Ideally, this data would be permanently stored for analysis and deeper research, like this. I used Qdrant because it’s an open source, has more content on the community, and fits better for this project, Research Matrix here.

I integrate OpenAI and Langchain to help create an interface that can interact with the Qdrant vector database. Langchain makes the connection between OpenAI’s API and the vector database easier, allowing more classes and functions.

I fine-tune Qdrant responses by adding context around user questions, enabling more informative answers while minimizing hallucinations. This approach structures responses to feel more conversational and aligned with user expectations.

I export the responses to evaluate their quality, refining them to sound more natural and human-like. I focus on clarity and conversational flow, with iterative testing to ensure that responses are accurate and consistently meet our quality standards.

In the last phase, Through DeepEval, I can measure the accuracy of responses and detect any deviations from factual information. 


