# rag_api/urls.py
from django.urls import path
from .views import IngestAPIView, QueryAPIView, query_retrieve_only, health

urlpatterns = [
    # endpoint 1: Ingest (POST only for file upload/ingestion)
    path("ingest/", IngestAPIView.as_view(), name="ingest"),
    # endpoint 2: Query (POST only for asking questions)
    path("query/", QueryAPIView.as_view(), name="query"),
    path("query_retrieve/", query_retrieve_only, name="query_retrieve"),
    path("health/", health, name="health"),
]
