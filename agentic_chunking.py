from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Tesla text to chunk
tesla_text = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.

Model Y Performance  
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.

Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs."""

prompt=f"""
You are a text chunking expert. Split this text into logical chunks.

Rules:
- Each chunk should be around 200 characters or less
- Split at natural topic boundaries
- Keep related information together
- Put "<<<SPLIT>>>" between chunks

Text:
{tesla_text}

Return the text with <<<SPLIT>>> markers where you want to split:
"""
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# Get the AI's response
print   ("Sending text to LLM for chunking...")

response = llm.invoke(prompt)
marked_text = response.content

#Split the text into chunks based on the <<<SPLIT>>> markers
chunks = [chunk.strip() for chunk in marked_text.split("<<<SPLIT>>>") if chunk.strip()]

# Print the resulting chunks
print("\n--- Chunks from LLM ---")
print("=" * 50)

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} --- ({len(chunk)} chars)")
    print(chunk)
    print("-" * 50)

