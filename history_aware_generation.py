from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)   

model = ChatOpenAI(model="gpt-4o")  

chat_history = []

def ask_question(user_query):
    print(f"\nUser Query: {user_query}")

    #step 1: Make the question clear using conversation history
    if chat_history:
        messages = [
            SystemMessage(content="You are a helpful assistant that provides clear and concise answers based on the user's query and the conversation history."),
            ]+chat_history+[
            HumanMessage(content=f"New question: {user_query}")
        ]
        result = model.invoke(messages)
        search_query = result.content.strip()
        print (f"Clarified Question: {search_query}")
    
    else:   
        search_query = user_query

    #step 2: Retrieve relevant documents from ChromaDB
    retriever = db.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(search_query)

    print(f"\nRetrieved {len(relevant_docs)} relevant documents:")
    for i, doc in enumerate(relevant_docs):
        print(f"\n--- Relevant Document {i+1} ---")
        print(f"Source: {doc.metadata['source']}")
        print(f"Length: {len(doc.page_content)} characters")
        print(f"Content:")
        print(doc.page_content)
        print("-" * 50)

    #step 3: Generate answer using LLM
    combined_input = f""" Based on the following documents, answer the user query: "{search_query}"

    Documents:  
    {chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."

    """
    #step 4: Generate answer using LLM
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on the provided documents."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    answer = result.content

    #step 5 : Update chat history
    chat_history.append(HumanMessage(content=user_query))
    chat_history.append(SystemMessage(content=answer))

    print(f"\n--- LLM Response ---\n{answer}")
    return answer

def start_chat():
    print("Welcome to the History-Aware Chatbot! Type 'exit' to quit.")
    while True:
        user_query = input("\nUser Query: ")
        if user_query.lower() == "exit":
            print("Exiting the chat. Goodbye!")
            break
        ask_question(user_query)

if __name__ == "__main__":
    start_chat()    