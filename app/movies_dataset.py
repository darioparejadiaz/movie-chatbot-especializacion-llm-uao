import pandas as pd
from sentence_transformers import SentenceTransformer
from ast import literal_eval


class MoviesDataset:

    # ***************************************

    def __init__(self, path):
        self.__path = path
        self.__model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.__data_frame = None
        self.__genres = None
        self.__preprocess_dataset()

    # ***************************************

    def __preprocess_dataset(self):
        self.__data_frame = pd.read_csv(self.__path)
        self.__data_frame = self.__data_frame.fillna(" ")
        self.__data_frame["Keywords"] = self.__data_frame["Plot Kyeword"].apply(
            self.__concatenate_list
        )
        self.__data_frame["Stars"] = self.__data_frame["Top 5 Casts"].apply(
            self.__concatenate_list
        )
        self.__data_frame["Generes"] = self.__data_frame["Generes"].apply(
            self.__string_to_list
        )
        self.__data_frame["Rating"] = (
            pd.to_numeric(self.__data_frame["Rating"], errors="coerce")
            .fillna(0)
            .astype("float")
        )
        self.__data_frame["text"] = self.__data_frame.apply(
            lambda x: str(x["Overview"]) + " " + x["Keywords"] + " " + x["Stars"],
            axis=1,
        )
        self.__data_frame.drop(["Plot Kyeword", "Top 5 Casts"], axis=1, inplace=True)

        embeddings = self.__model.encode(
            self.__data_frame["text"], batch_size=64, show_progress_bar=True
        )

        self.__data_frame["embeddings"] = embeddings.tolist()
        self.__data_frame["ids"] = self.__data_frame.index
        self.__data_frame["ids"] = self.__data_frame["ids"].astype("str")

        self.__genres = self.__data_frame["Generes"].explode().unique()

    # ***************************************

    def __concatenate_list(self, list):
        list = literal_eval(list)
        return " ".join(list)

    # ***************************************

    def __string_to_list(self, list):
        list = literal_eval(list)
        return list

    # ***************************************

    @property
    def data_frame(self):
        return self.__data_frame

    @property
    def genres(self):
        return list(self.__genres)
