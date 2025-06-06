# Vector Database & Semantic Search Pet Projects

This repository contains two pet projects designed to explore the world of vector databases and semantic search using Python. Each project tackles a different problem and uses a different vector database, demonstrating various concepts and technologies.

## Technologies Used

![Python](https://img.shields.io/badge/python-3.10+-blue.svg?logo=python&longCache=true&style=for-the-badge)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Pinecone](https://img.shields.io/badge/pinecone-008080.svg?style=for-the-badge&logo=pinecone&logoColor=white)
![Milvus](https://img.shields.io/badge/milvus-4fc4f9.svg?style=for-the-badge&logo=milvus&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

---

## Project 1: Smart Note Search

This project is a simple command-line application that allows you to save notes and search through them based on semantic meaning rather than exact keywords. For example, a search for "what to cook for dinner?" can find a note containing a "pasta recipe".

### ✨ Key Features

* **Semantic Search:** Finds notes based on meaning.
* **Secure API Key Handling:** Uses `.env` files for security.
* **Reproducible Environment:** Fully containerized with Docker.

### 🛠️ Technologies

* **Vector Database:** [Pinecone](https://www.pinecone.io/) (Cloud)
* **Embedding Model:** `all-MiniLM-L6-v2` from `sentence-transformers`

### 🚀 How to Run

1.  **Navigate to the project directory** (e.g., `/smart_notes`).

2.  **Set up environment variables:**
    Create a `.env` file in the project directory and add your Pinecone API key:
    ```env
    PINECONE_API_KEY="YOUR_API_KEY_HERE"
    ```

3.  **Run with Docker:**
    ```bash
    # Build the image
    docker build -t smart-notes-app .

    # Run the container using your API key directly
    docker run --rm -e PINECONE_API_KEY="YOUR_API_KEY_HERE" smart-notes-app
    ```

---

## Project 2: Movie Recommendation System

This project is a content-based recommendation system that suggests movies based on their genre similarity. Given a string of genres, it will find movies with the most semantically similar genre makeup.

### ✨ Key Features

* **Content-Based Recommendations:** Suggests similar items based on genre vectors.
* **Self-Hosted Vector DB:** Runs the production-grade Milvus vector database locally via Docker.
* **End-to-End Workflow:** Covers data cleaning, embedding, indexing, and searching.

### 🛠️ Technologies

* **Vector Database:** [Milvus](https://milvus.io/) (Self-hosted with Docker)
* **Data Handling:** Pandas
* **Embedding Model:** `all-MiniLM-L6-v2`

### 🚀 How to Run

1.  **Navigate to the project directory** (e.g., `/movie_recommender`). Ensure the `movies.csv` dataset is present in this directory.

2.  **Start the Milvus database:**
    ```bash
    docker-compose up -d
    ```

3.  **Build the application image:**
    ```bash
    docker build -t movie-recommender .
    ```

4.  **Run the application container:**
    The application needs to connect to the Milvus network. First, find the network name:
    ```bash
    docker network ls
    ```
    Look for a name like `movie_recommender_default` or `milvus`. Then, run the container, connecting it to that network:
    ```bash
    docker run --rm --network=movie_recommender_default movie-recommender
    ```

5.  **Shutdown Milvus (when finished):**
    To stop and remove the Milvus containers, run:
    ```bash
    docker-compose down
    ```

---

## License

This project is licensed under the MIT License.