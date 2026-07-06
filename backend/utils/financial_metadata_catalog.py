"""
Shared financial metadata vocabularies used at ingestion and query time.

Keeping these aliases in one module prevents the filename metadata extractor
and the query entity extractor from drifting apart over time.
"""

from backend.models.document import DocumentType

DOC_TYPE_ALIASES: dict[str, DocumentType] = {
    "annual report": DocumentType.ANNUAL_REPORT,
    "10-k": DocumentType.ANNUAL_REPORT,
    "10 k": DocumentType.ANNUAL_REPORT,
    "10k": DocumentType.ANNUAL_REPORT,
    "annual": DocumentType.ANNUAL_REPORT,
    "quarterly report": DocumentType.QUARTERLY_REPORT,
    "10-q": DocumentType.QUARTERLY_REPORT,
    "10 q": DocumentType.QUARTERLY_REPORT,
    "10q": DocumentType.QUARTERLY_REPORT,
    "quarterly": DocumentType.QUARTERLY_REPORT,
    "earnings call": DocumentType.EARNINGS_CALL,
    "earnings transcript": DocumentType.EARNINGS_CALL,
    "earnings": DocumentType.EARNINGS_CALL,
    "transcript": DocumentType.EARNINGS_CALL,
    "call": DocumentType.EARNINGS_CALL,
    "news": DocumentType.NEWS_ARTICLE,
    "news article": DocumentType.NEWS_ARTICLE,
    "article": DocumentType.NEWS_ARTICLE,
}

COMPANY_ALIASES: dict[str, tuple[str, str]] = {
    "apple": ("Apple Inc.", "AAPL"),
    "aapl": ("Apple Inc.", "AAPL"),
    "microsoft": ("Microsoft Corporation", "MSFT"),
    "msft": ("Microsoft Corporation", "MSFT"),
    "nvidia": ("NVIDIA Corporation", "NVDA"),
    "nvda": ("NVIDIA Corporation", "NVDA"),
    "tesla": ("Tesla, Inc.", "TSLA"),
    "tsla": ("Tesla, Inc.", "TSLA"),
    "amazon": ("Amazon.com, Inc.", "AMZN"),
    "amzn": ("Amazon.com, Inc.", "AMZN"),
    "alphabet": ("Alphabet Inc.", "GOOGL"),
    "google": ("Alphabet Inc.", "GOOGL"),
    "googl": ("Alphabet Inc.", "GOOGL"),
    "meta": ("Meta Platforms, Inc.", "META"),
    "meta platforms": ("Meta Platforms, Inc.", "META"),
    "meta platforms inc": ("Meta Platforms, Inc.", "META"),
    "meta platforms, inc.": ("Meta Platforms, Inc.", "META"),
    "netflix": ("Netflix, Inc.", "NFLX"),
    "nflx": ("Netflix, Inc.", "NFLX"),
    "salesforce": ("Salesforce, Inc.", "CRM"),
    "crm": ("Salesforce, Inc.", "CRM"),
}
