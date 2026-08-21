from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()


persistent_directory = "db/chroma_db"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

query = "What was NVIDIA's first graphics accelerator called?"

retriever = db.as_retriever(search_kwargs={"k": 3})

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
for i, doc in enumerate(relevant_docs):
    print(f"\n--- Relevant Document {i+1} ---")
    print(f"Source: {doc.metadata['source']}")
    print(f"Length: {len(doc.page_content)} characters")
    print(f"Content:")
    print(doc.page_content)
    print("-" * 50)


# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"

## LLM Chatgpt 
combined_input = f""" Based on the following documents, answer the user query: "{query}"

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."

"""


model= ChatOpenAI(model="gpt-4o")

messages = [
    SystemMessage(content="You are a helpful assistant that answers questions based on the provided documents."),
    HumanMessage(content=combined_input)
]

result = model.invoke(messages)

print(f"\n--- LLM Response ---\n{result.content}")
