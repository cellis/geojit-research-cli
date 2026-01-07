from __future__ import annotations

import uuid
from typing import Iterable
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue


def get_qdrant(qdrant_url: str | None = None, api_key: str | None = None) -> QdrantClient:
    if qdrant_url:
        return QdrantClient(url=qdrant_url, api_key=api_key)
    # local default
    return QdrantClient(host="127.0.0.1", port=6333)


def ensure_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    # If collection exists, leave it as-is; otherwise create.
    try:
        client.get_collection(name)
        return
    except Exception:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def upsert_points(client: QdrantClient, collection: str, ids: list[str], vectors, payloads: list[dict]) -> None:
    # Convert string IDs to UUIDs deterministically
    uuid_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, id_str)) for id_str in ids]
    # Use named vector for the collection
    points = [
        PointStruct(
            id=uuid_ids[i],
            vector={"geojit-dense-vector": vectors[i]},
            payload=payloads[i]
        )
        for i in range(len(ids))
    ]
    client.upsert(collection_name=collection, points=points)


def search(client: QdrantClient, collection: str, vector, top_k: int, filter_doc_id: int | None = None):
    query_filter = None
    if filter_doc_id is not None:
        query_filter = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=filter_doc_id))])
    return client.query_points(
        collection_name=collection,
        query=vector,
        using="geojit-dense-vector",
        limit=top_k,
        query_filter=query_filter
    ).points
