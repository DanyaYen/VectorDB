import os
import time
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from rich.console import Console
from dotenv import load_dotenv


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found. Please ensure you have a .env file with this value.")

INDEX_NAME = "smart-notes"


pc = Pinecone(api_key=PINECONE_API_KEY)
model = SentenceTransformer('all-MiniLM-L6-v2')
console = Console()


def setup_index():
    if INDEX_NAME not in pc.list_indexes().names():
        console.print(f"[yellow]Index '{INDEX_NAME}' not found. Creating a new one...[/yellow]")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  
            metric='cosine',  
            spec=ServerlessSpec(cloud='aws', region='us-east-1') 
        )
        console.print("[green]Index created successfully. Please wait a moment for it to initialize...[/green]")
        time.sleep(60)  
    else:
        console.print(f"[green]Index '{INDEX_NAME}' already exists.[/green]")
    
    return pc.Index(INDEX_NAME)

def populate_index(index):
    stats = index.describe_index_stats()
    if stats.total_vector_count == 0:
        my_notes = [
            "The best way to learn a new programming language is to build a pet project.",
            "Remember to buy milk and eggs on the way home.",
            "Carbonara pasta recipe: spaghetti, guanciale, eggs, pecorino cheese, black pepper.",
            "Vacation ideas: go to the mountains or relax by the sea.",
            "Quantum computers could revolutionize the world of computation."
        ]
        console.print("\n[yellow]Index is empty. Populating with notes...[/yellow]")
        
        vectors_to_upsert = []
        for i, note in enumerate(my_notes):
            embedding = model.encode(note).tolist()
            vectors_to_upsert.append({
                "id": f"note-{i+1}",
                "values": embedding,
                "metadata": {"text": note}
            })
        index.upsert(vectors=vectors_to_upsert, batch_size=100)
        console.print("[green]Notes added successfully![/green]")
    else:
        console.print("\n[cyan]Notes are already in the index. Skipping population.[/cyan]")
    
    console.print(index.describe_index_stats())

def search_notes(index, query, top_k=3):
    console.print(f"\n[bold cyan]Searching for:[/bold cyan] '{query}'")
    query_vector = model.encode(query).tolist()
    results = index.query(
        vector=query_vector, 
        top_k=top_k, 
        include_metadata=True
    )
    
    console.print("--- Search Results ---")
    if not results['matches']:
        console.print("[red]No matches found.[/red]")
        return
        
    for match in results['matches']:
        score = match['score']
        text = match['metadata']['text']
        console.print(f"[yellow]Similarity: {score:.4f}[/yellow] | [white]Text: {text}[/white]\n")


if __name__ == "__main__":
    pinecone_index = setup_index()
    populate_index(pinecone_index)
    
    console.print("\n[bold yellow]Waiting for 5 seconds for the index to update...[/bold yellow]")
    time.sleep(5)

    search_notes(pinecone_index, "What should I cook for dinner?")
    search_notes(pinecone_index, "How can I learn effectively?")
    search_notes(pinecone_index, "Where to go for a trip?")