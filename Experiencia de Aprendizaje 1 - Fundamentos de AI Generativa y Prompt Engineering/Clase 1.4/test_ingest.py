from src.ingesta.ingest import PDFIngester

ingester = PDFIngester(
    db_name="agent-rag-duoc-uc",
    collection_name="embeddings"
)

ingester.ingest_from_github(
    "https://github.com/Foco22/INGENIERIA-DE-SOLUCIONES-CON-INTELIGENCIA-ARTIFICIAL_002D_OLS"
)
