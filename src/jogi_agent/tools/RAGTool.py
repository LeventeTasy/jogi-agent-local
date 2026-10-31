import os
from typing import Type, ClassVar
from pydantic import BaseModel, Field, PrivateAttr
from crewai.tools import BaseTool
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rag import build_rag
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


class JogiKeresoInput(BaseModel):
    """Input schema for MyCustomTool."""
    query: str = Field(..., description="A jogi kérdés vagy keresőszavak.")


class RagTool(BaseTool):
    name: str = "Jogi adatbázis"
    description: str = (
        "Használd ezt a toolt, ha a Munka Törvénykönyve, Ptk vagy GDPR kapcsán kell keresned."
    )
    args_schema: Type[BaseModel] = JogiKeresoInput
    IS_BM25: ClassVar[bool] = False
    IS_HYBRID: ClassVar[bool] = True

    _db: Chroma = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #build_rag()

        # Elérési út, embeddings beállítása
        db_path = os.path.abspath(os.path.join(os.getcwd(), "../chroma_db"))
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

        # Mentés a belső _db változóba
        self._db = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
        print("--- RAG Adatbázis sikeresen betöltve a memóriába! ---")

    def _run(self, query: str) -> str:
        # A self._db-ből kérünk ki top 10 találatot)

        if self.IS_HYBRID:
            vector_retriever = self._db.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 10,
                    "filter": {"section_type": {"$in": ["cikk", "paragrafus"]}}
                }
            )

            data = self._db.get(include=["documents", "metadatas"])

            documents = [
                Document(page_content=text, metadata=metadata or {})
                for text, metadata in zip(data["documents"], data["metadatas"])
                if (metadata or {}).get("section_type") in ["cikk", "paragrafus"]
            ]

            bm25_retriever = BM25Retriever.from_documents(documents, k=10)

            ensemble = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.5, 0.5])

            results = [(doc, None) for doc in ensemble.invoke(query)]
        elif self.IS_BM25:
            data = self._db.get(include=["documents", "metadatas"])

            documents = [
                Document(page_content=text, metadata=metadata or {})
                for text, metadata in zip(data["documents"], data["metadatas"])
                if (metadata or {}).get("section_type") in ["cikk", "paragrafus"]
            ]

            retriever = BM25Retriever.from_documents(documents, k=10)

            results = [(doc, None) for doc in retriever.invoke(query)]
        else:
            results = self._db.similarity_search_with_score(query, k=10, filter={"section_type": {"$in": ["cikk", "paragrafus"]}})



        formatted_context = ""


        for doc, _score in results:
            meta = doc.metadata
            law = meta.get("law", "Ismeretlen törvény")
            sid = meta.get("section_id", "N/A")
            page = meta.get("page", "multiple")
            src = os.path.basename(meta.get("source", "file"))

            distance_score = "BM25" if _score is None else round(float(_score), 4)

            clean_content = doc.page_content[:1500] + "... [vágva a stabilitásért]" if len(
                doc.page_content) > 1500 else doc.page_content

            header = f"[Törvény: {law} | Hely: {sid} | Távolság: {distance_score} | Oldal: {page} | Forrásfájl: {src}]"
            formatted_context += f"{header}\n{clean_content}\n\n---\n\n"

        return formatted_context