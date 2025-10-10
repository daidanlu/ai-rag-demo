# main.py (RAGService CLI with Qdrant-friendly source printing)
import argparse
from rag import RAGService, DEFAULT_GPT4ALL_MODEL


def print_sources(hits, show_snippets=False):
    print("\n=== SOURCES ===")
    for i, h in enumerate(hits, 1):
        meta = h.get("meta") or {}
        doc = meta.get("doc_id", "?")
        idx = meta.get("chunk_idx", "?")
        print(f"[{i}] Document: {doc}, Chunk: {idx}")
        if show_snippets:
            text = (h.get("text") or "").strip().replace("\n", " ")
            if len(text) > 220:
                text = text[:220] + " ..."
            print(f"     ⤷ {text}")


def main():
    parser = argparse.ArgumentParser(description="AI RAG Demo CLI")
    sub = parser.add_subparsers(dest="cmd")

    # ingest
    p1 = sub.add_parser("ingest", help="Ingest PDFs into the RAG index")
    p1.add_argument(
        "patterns", nargs="+", help="Glob patterns for PDFs, e.g. data/*.pdf"
    )

    # ask
    p2 = sub.add_parser("ask", help="Ask a question against ingested documents")
    p2.add_argument("query", help="The question you want to ask")
    p2.add_argument(
        "--model", default=DEFAULT_GPT4ALL_MODEL, help="Override model name or file"
    )
    p2.add_argument("--k", type=int, default=4, help="Top-k results to retrieve")
    p2.add_argument(
        "--no-generate",
        action="store_true",
        help="Only retrieve; skip local LLM generation",
    )
    p2.add_argument(
        "--show-snippets",
        action="store_true",
        help="Print matched text snippets under Sources",
    )

    args = parser.parse_args()
    service = RAGService()

    if args.cmd == "ingest":
        service.ingest_files(args.patterns)
        return

    if args.cmd == "ask":
        hits = service.retrieve(args.query, k=args.k)
        if not hits:
            print("[No results found]")
            return

        if args.no_generate:
            print("\n=== ANSWER ===")
            print("[Generation skipped] Showing top-k retrieved chunks only.")
            print_sources(hits, show_snippets=args.show_snippets)
            return

        # Build prompt and run local LLM only when needed
        prompt = RAGService.build_prompt(args.query, hits)
        answer_text = RAGService.call_llamacpp(prompt, model_path=args.model)

        print("\n=== ANSWER ===")
        print(answer_text if answer_text else "[Empty answer]")
        print_sources(hits, show_snippets=args.show_snippets)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
