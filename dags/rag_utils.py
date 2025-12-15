import os
import json
import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import openai

# Configuration
CHROMA_DB_PATH = "/tmp/chroma_db"  # Use a persistent path in production
COLLECTION_NAME = "weather_data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class WeatherRAGSystem:
    def __init__(self, db_path=CHROMA_DB_PATH):
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def convert_record_to_text(self, record):
        """
        Converts a structured weather record into a descriptive text document.
        """
        # Handle timestamp if it's not in the record (it might be added during load)
        # For this pipeline, we'll assume we generate a timestamp or use the one provided
        timestamp = record.get('timestamp', datetime.datetime.now().isoformat())
        
        text = (
            f"Weather Report for Location (Lat: {record['latitude']}, Lon: {record['longitude']}) "
            f"on {timestamp}. "
            f"Temperature: {record['temperature']}°C. "
            f"Wind Speed: {record['windspeed']} km/h. "
            f"Wind Direction: {record['winddirection']}°. "
            f"Weather Code: {record['weathercode']}."
        )
        return text

    def embed_and_store(self, weather_data):
        """
        Embeds the weather data and stores it in ChromaDB.
        """
        text_doc = self.convert_record_to_text(weather_data)
        embedding = self.embedding_model.encode(text_doc).tolist()
        
        # Generate a unique ID (e.g., based on timestamp and location)
        # For simplicity, using a hash or current time if not unique enough
        record_id = f"{weather_data['latitude']}_{weather_data['longitude']}_{datetime.datetime.now().timestamp()}"
        
        metadata = {
            "latitude": weather_data['latitude'],
            "longitude": weather_data['longitude'],
            "temperature": weather_data['temperature'],
            "timestamp": datetime.datetime.now().isoformat(), # In a real scenario, use the data's timestamp
            "source": "open_meteo"
        }

        self.collection.add(
            documents=[text_doc],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[record_id]
        )
        print(f"Stored record {record_id} in ChromaDB.")
        return record_id

    def query_rag(self, query_text, n_results=3):
        """
        Retrieves relevant documents and generates an answer.
        """
        # 1. Embed the query
        query_embedding = self.embedding_model.encode(query_text).tolist()

        # 2. Retrieve from Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        retrieved_docs = results['documents'][0]
        retrieved_metadatas = results['metadatas'][0]

        # 3. Construct Context
        context = "\n\n".join(retrieved_docs)

        # 4. Generate Answer (using OpenAI as an example LLM)
        # Ensure OPENAI_API_KEY is set in environment
        try:
            client = openai.OpenAI()
            prompt = f"""
            You are a weather analyst. Use the following retrieved weather data to answer the user's question.
            If the answer is not in the context, say you don't know. Do not hallucinate.

            Context:
            {context}

            User Question: {query_text}

            Provide the response in the following JSON format:
            {{
                "retrieved_context_summary": "Summary of the weather data used",
                "model_reasoning": "Step-by-step reasoning based on data",
                "prediction_insight": "Final answer or prediction"
            }}
            """

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)

        except Exception as e:
            return {
                "error": str(e),
                "retrieved_context": context,
                "message": "Failed to generate response from LLM. Ensure OPENAI_API_KEY is set."
            }

# Helper function for Airflow task
def process_weather_embedding(weather_data):
    rag_system = WeatherRAGSystem()
    rag_system.embed_and_store(weather_data)
