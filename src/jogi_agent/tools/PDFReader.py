from pathlib import Path
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from pypdf import PdfReader


class PDFFullTextInput(BaseModel):
    """Input schema for PDFFullTextReaderTool."""
    file_path: str = Field(
        ...,
        description=(
            "A beolvasandó PDF fájl elérési útja az 'uploads/' mappában, "
            "pl. 'uploads/pelda.pdf'."
        ),
    )


class PDFFullTextReaderTool(BaseTool):
    name: str = "PDF Teljes Szöveg Beolvasó"
    description: str = (
        "Beolvassa egy adott PDF fájl TELJES szöveges tartalmát, oldalanként "
        "összefűzve. Csak azt a fájlt olvassa be, amit a file_path paraméterben megadsz."
    )
    args_schema: Type[BaseModel] = PDFFullTextInput

    def _run(self, file_path: str) -> str:
        path = Path(file_path)
        if not path.exists():
            return f"Hiba: a fájl nem található: {file_path}"

        reader = PdfReader(str(path))
        pages_text = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages_text.append(f"--- {i}. oldal ---\n{text}")

        return "\n\n".join(pages_text)