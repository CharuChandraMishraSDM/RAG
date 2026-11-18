# Services

Framework-independent building blocks of the ingestion pipeline. None of these
modules import FastAPI; the API layer (`app/api`) wires them together.

```
PDF bytes ──► pdf_service ──► ExtractedDocument ──► ChunkingService ──► list[DocumentChunk]
                              (pages + page numbers)                          │
                                                                              ▼
                                         ChunkEmbeddings ◄── EmbeddingService.embed_chunks
                                         (chunks + aligned unit vectors)
```

| Module | Responsibility |
| --- | --- |
| `pdf_service.py` | Validate an upload, extract text page by page with PyMuPDF |
| `chunking_service.py` | Normalize text and split each page into overlapping chunks |
| `embedding_service.py` | Turn chunk texts and queries into L2-normalized vectors |

## Chunking

`ChunkingService` uses LangChain's `RecursiveCharacterTextSplitter`. It tries
separators in order (`"\n\n"`, `"\n"`, `". "`, `" "`, `""`): it first splits on
paragraphs and only falls back to lines, sentences, words, and finally single
characters for pieces that are still longer than `chunk_size`. The pieces are
then merged back together up to `chunk_size` characters. Chunks therefore end at
the most natural boundary that fits.

Before splitting, `normalize_text` removes control characters, collapses runs of
spaces, tabs and non-breaking spaces, and limits blank lines to one. Newlines are
kept on purpose, because they are the splitter's preferred break points.

**Pages are chunked independently.** A chunk never spans two pages, so every
answer can cite one exact page. The trade-off is that a sentence crossing a page
break is split in two. The overlap within a page reduces the impact of this, but
does not remove it across pages.

### Why overlap?

Without overlap, a fact that straddles a chunk boundary is cut in half, and
neither half may be similar enough to the question to be retrieved. With
`chunk_overlap=150`, the last ~150 characters of one chunk are repeated at the
start of the next, so a sentence of up to that length near a boundary appears
whole in at least one chunk. The cost is about 20% more chunks to embed and
store, plus some duplicate text in the retrieved context.

The splitter measures overlap in whole pieces (words or sentences), not exact
characters, so the actual overlap is *at most* `chunk_overlap` characters.

### Why 800 / 150 is only a starting point

- **Smaller chunks** give more precise matches and citations, but each chunk
  carries less context, so the LLM may see a fact without its explanation.
- **Larger chunks** carry more context, but their single vector averages several
  topics, which makes it less similar to any one question.
- The embedding model truncates input longer than **256 tokens**. 800 characters
  of English is roughly 150–200 tokens, which fits. Much larger chunks would be
  silently truncated before they are embedded.

The right values depend on the documents (dense runbooks vs. narrative reports)
and the questions people ask. Tune them by measuring retrieval quality on real
questions; both are configurable through `CHUNK_SIZE` and `CHUNK_OVERLAP`.

## Embeddings

`all-MiniLM-L6-v2` is a 6-layer transformer fine-tuned so that texts with similar
meaning get nearby vectors. For each text, the model:

1. tokenizes it into word pieces (up to 256),
2. produces a contextual vector for every token,
3. **mean-pools** the token vectors into one fixed-size sentence vector.

**Why 384 dimensions?** It is the model's hidden size, a design choice of
MiniLM that keeps it small (~90 MB) and fast on CPU. The service never hardcodes
384. It reads the size from `model.get_embedding_dimension()` and checks every
batch's output shape against it. The integration tests confirm that the real
model reports 384.

`EmbeddingService` loads the model **once**, when it is constructed. The API
keeps one instance per process (`get_embedding_service` is `lru_cache`d), and the
model loads on the first upload rather than at server startup. Texts are encoded
in batches (`EMBEDDING_BATCH_SIZE`, default 32), which is much faster than
encoding them one at a time.

### L2 normalization, cosine similarity and FAISS

Cosine similarity compares direction and ignores length:

```
cos(a, b) = (a · b) / (‖a‖ ‖b‖)
```

If every vector is first scaled to length 1 (L2 normalization, `v / ‖v‖`), the
denominator is 1 and **cosine similarity equals the plain inner product
`a · b`**. That matters for the next milestone: FAISS's `IndexFlatIP` ranks by
inner product. Because chunk vectors and query vectors are both normalized, an
inner-product search is an exact cosine-similarity search, and scores fall in
[-1, 1] where higher means more similar.

For unit vectors, Euclidean distance is also determined by cosine similarity
(`‖a − b‖² = 2 − 2·cos(a, b)`), so an L2 index would give the same ranking. Inner
product is simply the more direct choice.

Normalization is done in `l2_normalize` rather than through the library flag, so
it is explicit and unit-testable. It also guards against zero vectors (no NaNs).
Output is always `float32`, the type FAISS expects, with shape `(n, dimension)`
for batches and `(dimension,)` for a query.

## Metadata preservation

Metadata travels with the text at every stage:

- `PageText.page_number` is 1-based, as users see pages. Blank pages are kept
  during extraction so later page numbers never shift.
- Every `DocumentChunk` carries `document_id`, `filename`, `page_number`,
  `chunk_index` and `text`. `chunk_index` is sequential across the whole
  document, so `(document_id, chunk_index)` uniquely identifies a chunk.
- `embed_chunks` returns `ChunkEmbeddings`, which pairs `chunks[i]` with
  `vectors[i]` and refuses to be built if the counts differ. The vector store can
  use the row position as the FAISS id and look the metadata up by the same
  position.

## Example

```python
from pathlib import Path

from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.pdf_service import extract_document

pdf_bytes = Path("database_runbook.pdf").read_bytes()
document = extract_document("database_runbook.pdf", pdf_bytes, max_size_bytes=20 * 1024 * 1024)

chunks = ChunkingService(chunk_size=800, chunk_overlap=150).chunk_document(document)

embedder = EmbeddingService()  # loads all-MiniLM-L6-v2 once, on CPU
embedded = embedder.embed_chunks(chunks)
query = embedder.embed_query("How is the connection pool configured?")

scores = embedded.vectors @ query          # cosine similarities, shape (len(chunks),)
best = embedded.chunks[int(scores.argmax())]
print(best.page_number, best.chunk_index, best.text[:80])
```

This brute-force `@` comparison is exactly what FAISS will do for us, at scale,
in the vector store milestone.

## Known limitations

- Scanned (image-only) PDFs have no text layer and are rejected; OCR is not
  implemented.
- Text order follows PyMuPDF's extraction order, so multi-column layouts and
  tables may be read in an unnatural order.
- Hyphenated words broken across lines are not rejoined.
