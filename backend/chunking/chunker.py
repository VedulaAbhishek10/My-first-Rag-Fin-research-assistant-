"""
Text chunker — splits document text into overlapping pieces.

Why chunk?
  Embedding models have a maximum input length (usually 512 tokens ≈ ~380 words).
  A 100-page SEC filing is far too long to embed as one piece. Chunking splits
  it into small pieces that each fit inside the model's context window.

Why overlap?
  Without overlap, a sentence that straddles two chunk boundaries is split in
  half. Each half loses the context of the other half. Overlap repeats the
  last `chunk_overlap` characters of the previous chunk at the start of the
  next one, so no sentence ever loses its surrounding context.

Strategy used here: character-level sliding window with word-boundary snapping.
  Simple, fast, and dependency-free. Works well for financial prose.
"""

from dataclasses import dataclass


@dataclass
class TextChunk:
    """A single chunk of text from a document page."""

    text: str
    page_number: int | None
    char_start: int  # character offset in the original page text


class TextChunker:
    """Splits text into overlapping fixed-size chunks."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        """
        Args:
            chunk_size:    Maximum characters per chunk.
            chunk_overlap: Characters shared between adjacent chunks.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, page_number: int | None = None) -> list[TextChunk]:
        """
        Split a single text string into overlapping chunks.

        Args:
            text:        The raw text to split.
            page_number: The source page number to attach to each chunk.

        Returns:
            A list of TextChunk objects. Empty list if text is blank.
        """
        text = text.strip()
        if not text:
            return []

        # Short text fits in one chunk — no splitting needed
        if len(text) <= self.chunk_size:
            return [TextChunk(text=text, page_number=page_number, char_start=0)]

        chunks: list[TextChunk] = []
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            # Snap to the nearest word boundary so we don't cut mid-word.
            # Only snap if we're not at the very end of the text.
            if end < len(text):
                last_space = chunk_text.rfind(" ")
                if last_space > self.chunk_size // 2:
                    end = start + last_space
                    chunk_text = text[start:end]

            chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        page_number=page_number,
                        char_start=start,
                    )
                )

            # Step forward by (chunk_size - overlap) to create the overlap window
            step = max(1, (end - start) - self.chunk_overlap)
            start += step

        return chunks

    def chunk_pages(self, pages: list[tuple[int, str]]) -> list[TextChunk]:
        """
        Chunk multiple pages independently, preserving page numbers.

        Args:
            pages: List of (page_number, text) tuples.

        Returns:
            All chunks from all pages in order.
        """
        all_chunks: list[TextChunk] = []
        for page_number, text in pages:
            all_chunks.extend(self.chunk(text, page_number=page_number))
        return all_chunks
