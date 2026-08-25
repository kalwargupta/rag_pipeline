import chromadb

# 1. Connect to your existing local database directory
# Replace "./chroma_db" with the actual path to your database folder
client = chromadb.PersistentClient(path="./db/chroma_db")

# 2. List all available collections to find your target name
print("Available collections:", client.list_collections())

# 3. Get your specific collection
# Replace "my_collection" with your actual collection name
collection = client.get_collection(name="my_collection")

# 4. Fetch the data AND explicitly request the raw vectors
# ChromaDB requires include=['embeddings'] to actually output the numbers
data = collection.get(include=['embeddings', 'documents', 'metadatas'])

# 5. Loop through and inspect the vectors
for i in range(len(data['ids'])):
    doc_id = data['ids'][i]
    document = data['documents'][i] if data['documents'] else "No text content"
    vector = data['embeddings'][i]
    
    print(f"\n--- Item {i+1} ---")
    print(f"ID: {doc_id}")
    print(f"Document Text: {document}")
    print(f"Vector Dimensions: {len(vector)}")
    print(f"First 5 Vector Values: {vector[:5]}...") 
