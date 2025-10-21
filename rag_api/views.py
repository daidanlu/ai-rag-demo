# rag_api/views.py
import os
import time
import requests
from rag import RAGService
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rag_backend.storage_factory import clear_storage
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from pathlib import Path

from .serializers import IngestSerializer, QuerySerializer

rag_service = RAGService()


class IngestAPIView(APIView):
    """
    API endpoint for uploading PDF files and triggering the RAG ingestion process.
    """

    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        serializer = IngestSerializer(data=request.data)

        # 1. ensure file is present
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        uploaded_file = serializer.validated_data["pdf_file"]

        # 2. save the uploaded file to the data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True, parents=True)

        file_path = data_dir / uploaded_file.name

        try:
            # handle the file in chunks to prevent large memory consumption
            with open(file_path, "wb+") as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            # 3. call the core RAG ingestion logic
            chunks_processed = rag_service.ingest_files([str(file_path)])

            if chunks_processed > 0:
                return Response(
                    {
                        "status": "success",
                        "message": "File ingested successfully.",
                        "chunks_processed": chunks_processed,
                        "document_id": uploaded_file.name,
                    },
                    status=status.HTTP_201_CREATED,
                )
            else:
                return Response(
                    {
                        "status": "error",
                        "message": "File uploaded but no chunks were extracted (empty PDF?).",
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

        except Exception as e:
            return Response(
                {
                    "status": "error",
                    "message": "Internal server error during processing.",
                    "details": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class QueryAPIView(APIView):
    """
    API endpoint for asking questions against ingested documents.
    """

    parser_classes = (JSONParser,)

    def post(self, request, *args, **kwargs):
        serializer = QuerySerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        validated = serializer.validated_data

        # use bool to handle true/false/1/0/"true"/"false"
        gen = validated.get("generate", True)
        if isinstance(gen, str):
            gen = gen.strip().lower() in ("1", "true", "yes", "y")

        try:
            result = rag_service.answer(
                query=validated["query"],
                k=validated.get("k", 4),
                generate=gen,
            )

            if result.get("answer", "").startswith("[ERROR]"):
                return Response(
                    {
                        "status": "error",
                        "message": "LLM generation failed. Check model file or llama-cpp installation.",
                        "answer": result.get("answer", ""),
                        "sources": result.get("hits", []),
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

            return Response(
                {
                    "status": "success",
                    "answer": result.get("answer", "No answer generated."),
                    "sources": result.get("hits", []),
                },
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            return Response(
                {
                    "status": "error",
                    "message": "Internal server error during query processing.",
                    "details": str(e),
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


@api_view(["POST"])
def query_retrieve_only(request):
    try:
        query = (request.data or {}).get("query", "")
        k = int((request.data or {}).get("k", 4))
        if not query:
            return Response(
                {"status": "error", "message": "Field 'query' is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        hits = rag_service.retrieve(query, k=k)

        return Response(
            {"status": "success", "sources": hits}, status=status.HTTP_200_OK
        )

    except Exception as e:
        return Response(
            {"status": "error", "message": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


# health & config check
@api_view(["GET"])
def health(request):
    """
    Returns current storage backend and (if Qdrant) a quick connectivity check.
    """
    backend = os.getenv("RAG_STORAGE", "memory").lower()
    info = {"status": "ok", "storage": backend}

    if backend == "qdrant":
        import requests

        url = os.getenv("QDRANT_URL", "http://127.0.0.1:6333").rstrip("/")
        coll = os.getenv("QDRANT_COLLECTION", "chunks")
        info["qdrant"] = {"url": url, "collection": coll}

        try:
            start = time.time()
            # 1) ping server root (fast)
            r0 = requests.get(f"{url}/", timeout=5)
            server_ok = r0.status_code == 200

            # 2) check collection (also returns points_count)
            r1 = requests.get(f"{url}/collections/{coll}", timeout=5)
            collection_ok = r1.status_code == 200
            points_count = None
            if collection_ok:
                try:
                    points_count = r1.json().get("result", {}).get("points_count")
                except Exception:
                    pass

            info["qdrant"]["server_ok"] = server_ok
            info["qdrant"]["collection_ok"] = collection_ok
            info["qdrant"]["points_count"] = points_count
            info["qdrant"]["alive"] = bool(server_ok and collection_ok)
            info["latency_ms"] = round((time.time() - start) * 1000, 2)

        except Exception as e:
            info["qdrant"]["alive"] = False
            info["qdrant"]["error"] = str(e)

    return Response(info, status=status.HTTP_200_OK)


# POST /api/v1/clear/, drops and recreates the vector index (Qdrant) or clears memory store.
class ClearAPIView(APIView):
    def post(self, request):
        result = clear_storage()
        status_txt = "success" if result.get("ok") else "failed"
        return Response({"status": status_txt, "result": result})
