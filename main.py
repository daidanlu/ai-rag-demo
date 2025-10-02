import argparse
from rag import ingest_files, retrieve, build_prompt, call_llamacpp

def main():
    parser = argparse.ArgumentParser(description="AI RAG Demo CLI")
    sub = parser.add_subparsers(dest="cmd")

    p1 = sub.add_parser("ingest", help="Ingest PDFs into Chroma DB")
    p1.add_argument("patterns", nargs="+", help="Glob patterns for PDFs, e.g. data/*.pdf")

    p2 = sub.add_parser("ask", help="Ask a question against ingested documents")
    p2.add_argument("query", help="The question you want to ask")
    p2.add_argument("--model", default=None, help="Override model name or file")
    p2.add_argument("--k", type=int, default=4, help="Top-k results to retrieve")

    args = parser.parse_args()

    if args.cmd == "ingest":
        ingest_files(args.patterns)

    elif args.cmd == "ask":
        hits = retrieve(args.query, k=args.k)
        if not hits:
            print("[No results found]")
            return
        prompt = build_prompt(args.query, hits)
        model = args.model if args.model else None
        answer = call_llamacpp(prompt, model_path=model or None)
        print("\n=== ANSWER ===")
        print(answer)
        print("\n=== SOURCES ===")
        for i, h in enumerate(hits, 1):
            print(f"[{i}] {h['meta']}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
