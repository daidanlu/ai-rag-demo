# rag_api/views.py
import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from .serializers import IngestSerializer
from pathlib import Path
from rag import RAGService
from .serializers import QuerySerializer
from rest_framework.parsers import JSONParser

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
            # pass the file path as a pattern list for the ingest_files method
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
            # explicitly returning the error string in the response
            error_detail = str(e)
            return Response(
                {
                    "status": "error",
                    "message": "Internal server error during processing.",
                    "details": error_detail,
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class QueryAPIView(APIView):
    """
    API endpoint for asking questions against ingested documents.
    """

    # explicitly specify the parser
    parser_classes = (JSONParser,)

    def post(self, request, *args, **kwargs):
        serializer = QuerySerializer(data=request.data)

        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        validated_data = serializer.validated_data

        try:
            # 1. calling the answer core logic of RAG service
            result = rag_service.answer(
                query=validated_data["query"],
                k=validated_data.get("k", 4),
                generate=validated_data.get("generate", True),
            )

            # 2. handle LLM error fallback, if LLM returns an error string, return status code 500
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

            # 3. success
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
