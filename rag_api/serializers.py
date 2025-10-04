# rag_api/serializers.py
from rest_framework import serializers


class IngestSerializer(serializers.Serializer):
    """
    Serializer for handling PDF file upload during ingestion.
    """

    pdf_file = serializers.FileField(required=True, help_text="PDF file to ingest.")


class QuerySerializer(serializers.Serializer):
    """
    Serializer for handling user queries.
    """

    # core query
    query = serializers.CharField(
        max_length=500,
        required=True,
        help_text="The question to ask against ingested documents.",
    )
    # optional parameter k and generate
    k = serializers.IntegerField(default=4, help_text="Top-K chunks to retrieve.")
    generate = serializers.BooleanField(
        default=True,
        help_text="If true, use LLM to generate answer; otherwise, return chunks.",
    )
