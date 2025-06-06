import pandas as pd
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer

DATASET_PATH = "movies.csv" 
NUM_MOVIES_TO_PROCESS = 5000

MILVUS_ALIAS = "default"
MILVUS_HOST = "milvus-standalone"
MILVUS_PORT = "19530"

COLLECTION_NAME = "movie_recommender"
EMBEDDING_DIM = 384

def prepare_data():
    df = pd.read_csv(DATASET_PATH)
    df.dropna(subset=['title', 'genre'], inplace=True)
    df.drop_duplicates(subset=['title'], inplace=True)
    df = df.head(NUM_MOVIES_TO_PROCESS)
    df.reset_index(drop=True, inplace=True)
    return df

def connect_to_milvus():
    connections.connect(alias=MILVUS_ALIAS, host=MILVUS_HOST, port=MILVUS_PORT)

def create_milvus_collection():
    if utility.has_collection(COLLECTION_NAME):
        return
    fields = [
        FieldSchema(name="movieId", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="genre", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    ]
    schema = CollectionSchema(fields=fields, description="Movie recommendation collection")
    Collection(name=COLLECTION_NAME, schema=schema)

def embed_and_insert_data(movie_data):
    collection = Collection(name=COLLECTION_NAME)
    if collection.num_entities > 0:
        return
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    genre_embeddings = embedding_model.encode(
        movie_data['genre'].tolist(), 
        show_progress_bar=True
    )
    data_to_insert = [
        movie_data['movieId'].tolist(),
        movie_data['title'].tolist(),
        movie_data['genre'].tolist(),
        genre_embeddings
    ]
    collection.insert(data_to_insert)
    collection.flush()

def create_index_and_load():
    collection = Collection(name=COLLECTION_NAME)
    if not collection.has_index():
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()

def search_similar_movies(query_genre: str, top_k: int = 5):
    collection = Collection(name=COLLECTION_NAME)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    query_vector = embedding_model.encode(query_genre)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["title", "genre"]
    )
    print(f"\nRecommendations for genre '{query_genre}':")
    for hit in results[0]:
        print(f"  - Title: {hit.entity.get('title')}, Genre: {hit.entity.get('genre')}, Distance: {hit.distance:.4f}")

if __name__ == "__main__":
    movie_data = prepare_data()
    connect_to_milvus()
    create_milvus_collection()
    embed_and_insert_data(movie_data)
    create_index_and_load()
    
    search_similar_movies(query_genre="Action|Thriller|Sci-Fi")
    search_similar_movies(query_genre="Animation|Children|Comedy")