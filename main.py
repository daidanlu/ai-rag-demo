# main.py (Updated to use RAGService)
import argparse
from rag import RAGService, DEFAULT_GPT4ALL_MODEL


def main():
    parser = argparse.ArgumentParser(description="AI RAG Demo CLI")
    sub = parser.add_subparsers(dest="cmd")

    p1 = sub.add_parser("ingest", help="Ingest PDFs into the RAG index")
    p1.add_argument(
        "patterns", nargs="+", help="Glob patterns for PDFs, e.g. data/*.pdf"
    )

    p2 = sub.add_parser("ask", help="Ask a question against ingested documents")
    p2.add_argument("query", help="The question you want to ask")
    p2.add_argument(
        "--model", default=DEFAULT_GPT4ALL_MODEL, help="Override model name or file"
    )
    p2.add_argument("--k", type=int, default=4, help="Top-k results to retrieve")

    args = parser.parse_args()

    service = RAGService()

    if args.cmd == "ingest":
        service.ingest_files(args.patterns)

    elif args.cmd == "ask":
        hits = service.retrieve(args.query, k=args.k)
        if not hits:
            print("[No results found]")
            return

        prompt = RAGService.build_prompt(args.query, hits)

        # Use args.model (which defaults to DEFAULT_GPT4ALL_MODEL if not overridden)
        answer_text = RAGService.call_llamacpp(prompt, model_path=args.model)

        print("\n=== ANSWER ===")
        print(answer_text)
        print("\n=== SOURCES ===")
        for i, h in enumerate(hits, 1):
            # Print the text snippet from the hit
            print(
                f"[{i}] Document: {h['meta'].get('doc_id','?')}, Chunk: {h['meta'].get('chunk_idx','?')}"
            )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
