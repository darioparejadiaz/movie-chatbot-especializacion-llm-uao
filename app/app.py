import os
from dotenv import load_dotenv
import openai
from movies_dataset import MoviesDataset
from pinecone_vector_database import PineconeVectorDatabase
from chat_ui import ChatUI


def main():
    load_dotenv()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = openai_api_key

    movies_dataset = MoviesDataset(path="../data/25k-IMDb-movie-Dataset.csv")
    movie_genres = movies_dataset.genres
    movies_data_frame = movies_dataset.data_frame
    pinecone_vector_database = PineconeVectorDatabase(
        api_key=pinecone_api_key, data_frame=movies_data_frame
    )

    ChatUI(llm=openai, search_fn=pinecone_vector_database.search, genres=movie_genres)


if __name__ == "__main__":
    main()
