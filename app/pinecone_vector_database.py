import pinecone
from pinecone import Pinecone
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer


class PineconeVectorDatabase:

    # ***************************************

    def __init__(self, api_key, data_frame):
        self.__index_name = "movies-embeddings"
        self.__dimension_embeddings = 384
        self.__index = None
        self.__batch_size = 64
        self.__model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.__pc = Pinecone(api_key=api_key, environment="asia-southeast1-gcp-free")
        if self.__create_index():
            self.__create_embeddings(data_frame)

    # ***************************************

    def __create_index(self):
        all_index_names = self.__pc.list_indexes()

        if self.__index_name not in all_index_names.names():
            print(f"El índice '{self.__index_name}' no existe. Creándolo ahora.")
            self.__pc.create_index(
                self.__index_name,
                dimension=self.__dimension_embeddings,
                metric="cosine",
                spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1"),
            )

            self.__index = self.__pc.Index(self.__index_name)
            return True
        else:
            print(
                f"El índice '{self.__index_name}' ya existe. Conectando al índice existente."
            )

            self.__index = self.__pc.Index(self.__index_name)
            return False

    # ***************************************

    def __create_embeddings(self, data_frame):
        for i in tqdm(range(0, len(data_frame), self.__batch_size)):
            i_end = min(i + self.__batch_size, len(data_frame))
            batch = data_frame[i:i_end]
            ids = batch["ids"]
            emb = batch["embeddings"]
            metadata = batch.drop(
                ["ids", "embeddings", "text", "path"], axis=1
            ).to_dict("records")
            to_upsert = list(zip(ids, emb, metadata))
            self.__index.upsert(vectors=to_upsert)

    # ***************************************

    def search(self, query, genre, rating, top_k):
        query_vector = self.__model.encode(query).tolist()

        if rating:
            filter_rating = rating
        else:
            filter_rating = 0

        if genre:
            conditions = {
                "Generes": {"$in": [genre]},
                "Rating": {"$gte": filter_rating},
            }
        else:
            conditions = {
                "Rating": {"$gte": filter_rating},
            }

        responses = self.__index.query(
            vector=query_vector, top_k=top_k, include_metadata=True, filter=conditions
        )

        response_data = []
        for response in responses["matches"]:
            response_data.append(
                {
                    "Title": response["metadata"]["movie title"],
                    "Overview": response["metadata"]["Overview"],
                    "Director": response["metadata"]["Director"],
                    "Genre": response["metadata"]["Generes"],
                    "year": response["metadata"]["year"],
                    "Rating": response["metadata"]["Rating"],
                    "Score": response["score"],
                }
            )

        data_frame = pd.DataFrame(response_data)
        return data_frame
