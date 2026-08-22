from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

tesla_text = """Tesla's Q3 Results

Tesla reported record revenue of $25.2B in Q3 2024.

Model Y Performance

The Model Y became the best-selling vehicle globally, with 350,000 units sold.

Production Challenges

Supply chain issues caused a 12% increase in production costs.

This is one very long paragraph that definitely exceeds our 100 character limit and has no double newlines inside it whatsoever making it impossible to split properly."""


splitter1 = CharacterTextSplitter(
    #seperators="\n\n", 
    chunk_size=100, 
    chunk_overlap=0
)

chunks1 = splitter1.split_text(tesla_text)

for i, chunk in enumerate(chunks1):
    print(f"--- Chunk {i+1} --- ({len(chunk)} chars)")
    print(chunk)
    print("-" * 50)

print("\n" + "=" * 50)
print("\n\nUsing RecursiveCharacterTextSplitter:\n")
print("=" * 50) 

splitter2 = RecursiveCharacterTextSplitter(
    chunk_size=100, 
    chunk_overlap=0,
    separators=["\n\n", "\n", " ", ""]

)

chunks2 = splitter2.split_text(tesla_text)

for i,chunk in enumerate (chunks2):
    print(f"--- Chunk {i+1} --- ({len(chunk)} chars)")
    print(chunk)
    print("-" * 50)