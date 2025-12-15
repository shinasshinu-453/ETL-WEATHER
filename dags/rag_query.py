import sys
import os
import json

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_utils import WeatherRAGSystem

def main():
    print("Initializing RAG System...")
    rag = WeatherRAGSystem()
    
    # Example Query 1
    query1 = "Will rainfall increase in Chennai next week based on historical trends?"
    print(f"\nQuery: {query1}")
    
    # Note: In a real run, this requires data to be in ChromaDB and OPENAI_API_KEY to be set.
    # If running without data, it might return empty context.
    try:
        response1 = rag.query_rag(query1)
        print("Response:")
        print(json.dumps(response1, indent=2))
    except Exception as e:
        print(f"Error executing query: {e}")

    print("-" * 50)

    # Example Query 2
    query2 = "Analyze abnormal temperature patterns in the last 30 days"
    print(f"Query: {query2}")
    try:
        response2 = rag.query_rag(query2)
        print("Response:")
        print(json.dumps(response2, indent=2))
    except Exception as e:
        print(f"Error executing query: {e}")

if __name__ == "__main__":
    main()
