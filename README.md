# 🎢 Retrieval-Augmented Generation (RAG) Chatbot with DeepEval Evaluation – Disney Chat 🧑‍🚀

![Disney Chatbot](https://github.com/roangws/disney-chat/blob/main/chat.png)

Welcome to the **Disney Chatbot**! Interact with live data for Disney and other global theme park attractions! 🌍

---

## 📌 Project Overview

This project showcases a **Retrieval-Augmented Generation (RAG) chatbot** designed to generate human-like responses based on real-time data from worldwide theme parks. The chatbot uses:
- **Langchain** and **Qdrant** for vector-based data retrieval and generation.
- **OpenAI GPT-3.5-turbo** for natural language generation.
- **DeepEval Evaluation** to assess response quality, relevance, and factuality.

## 🚀 Implementation Steps

### 1. Data Collection from ThemeParks API 🌐
Started by connecting to the **ThemeParks API** to gather data on various theme parks and their attraction wait times. Each park has a unique ID for accessing detailed attraction data, which is exported to an Excel file for easier visualization.

### 2. Data Integration with Qdrant 🗄️
The retrieved data is stored in the **Qdrant** vector database. Any pre-existing data is cleared and replaced with the latest data for testing purposes. In a full implementation, functions would be created to detect and handle data updates automatically. **Qdrant** was chosen for its open-source model and active community, making it ideal for this project.

### 3. Generative AI with OpenAI & Langchain 🤖
Using **Langchain**, I seamlessly connect **OpenAI’s API** with Qdrant, enabling complex functions and classes. This setup allows for smoother data retrieval and more effective RAG-based conversations.

### 4. Enhancing Responses with Context 📝
I refined responses by embedding contextual information around user queries. This process enhances answer quality, creating conversational, informative responses while reducing potential AI hallucinations.

### 5. Response Quality Assessment with DeepEval ✅
In the final stage, **DeepEval** is used to measure response accuracy, relevancy, and detect any factual inaccuracies. Responses are iteratively adjusted to sound more human-like, prioritizing clarity and conversational tone.

---

## ⚙️ Future Directions
This chatbot architecture provides a foundation for future enhancements, such as:
- Predictive routing and user journey optimization.
- Advanced wait time forecasting based on live data.

---

🎉 **Thank you for exploring this project!** Feel free to dive into the code and reach out with any questions.

--- 

> **Icons** sourced from [Font Awesome](https://fontawesome.com/) and [Material Icons](https://material.io/resources/icons/)
