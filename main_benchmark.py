import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointIdsList
from qdrant_client.http.models import VectorParams, Distance, SearchParams, PointIdsList
import faiss
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text

# ------------------------------------
# Config
# ------------------------------------
EMBEDDINGS_FILE = "cifar10_embeddings.npy"
QUERY_COUNT = 5
TOP_K = 5
DIM = 512 # CLIP embedding dimension
SIZES = [100, 1000, 5000, 10000]

# ------------------------------------
# Load Embeddings
# ------------------------------------
print("Loading embeddings...")
all_embeddings = np.load(EMBEDDINGS_FILE)
MAX_DATA = min(10000, len(all_embeddings))
embeddings = all_embeddings[:MAX_DATA]
queries = embeddings[:QUERY_COUNT]
N = embeddings.shape[0]
print(f"Loaded {N} embeddings of dimension {DIM}")

# ------------------------------------
# Compute ground truth with FAISS FlatL2
# ------------------------------------
def compute_ground_truth(size):
    print(f"Building FAISS flat index (ground truth)... for {size} embeddings")
    sub_embeddings = embeddings[:size]
    faiss_flat = faiss.IndexFlatL2(DIM)
    faiss_flat.add(sub_embeddings)
    gt_distances, gt_indices = faiss_flat.search(queries, TOP_K)
    return gt_indices

# ------------------------------------
# Recall calculation
# ------------------------------------
def recall_at_k(results, ground_truth):
    if len(results) != len(ground_truth):
        raise ValueError(f"Length mismatch: results={len(results)}, ground_truth={len(ground_truth)}")
    match_count = 0
    total_count = len(ground_truth) * TOP_K
    for i in range(len(ground_truth)):
        match_count += len(set(results[i]).intersection(set(ground_truth[i])))
    return match_count / total_count

# ------------------------------------
# Milvus Benchmark
# ------------------------------------
def benchmark_milvus():
    print("\n--- Benchmarking Milvus ---")
    results = []
    connections.connect("default", host="127.0.0.1", port="19530")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="label", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM)
    ]
    schema = CollectionSchema(fields, description="CIFAR embeddings")

    for size in SIZES:
        print(f"\nTesting size = {size}...")

        # <<<<< Table creation >>>>>
        start_setup = time.time()
        if utility.has_collection("cifar"):
            utility.drop_collection("cifar")
        collection = Collection(name="cifar", schema=schema)
        setup_time = time.time() - start_setup

        # Data
        sub_embeddings = embeddings[:size]
        sub_labels = np.random.randint(0, 10, size)
        ids = list(range(size))

        # <<<<< Inserts >>>>>
        print(f"Inserting with size: {size}...")
        start_insert = time.time()
        collection.insert([ids, sub_labels, sub_embeddings.tolist()])
        insert_time = time.time() - start_insert
        collection.flush()

        # Create index - Use FLAT for small sizes to avoid IVF recall issues
        index_type = "IVF_FLAT" if size >= 1000 else "FLAT"
        collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": index_type,
                "metric_type": "L2",
                "params": {"nlist": min(128, size // 2)} if index_type == "IVF_FLAT" else {}
            }
        )

        collection.load()
        gt_indices = compute_ground_truth(size)
        milvus_results = []
        total_topk_time, total_filter_time, total_range_time = 0, 0, 0

        for qi in range(QUERY_COUNT):
            print(f"Querying functions - Loop {qi+1}")
            query_vec = queries[qi].tolist()

            # <<<<< Query - Top-K search >>>>>
            start_query_topk = time.time()
            topk_hits = collection.search([query_vec], "embedding", param={"metric_type": "L2"}, limit=TOP_K)
            total_topk_time += time.time() - start_query_topk
            milvus_results.append([hit.id for hit in topk_hits[0]])

            # <<<<< Query - Top-K search with filter (label = 1) >>>>>
            start_query_filter = time.time()
            _ = collection.search([query_vec], "embedding", param={"metric_type": "L2"}, limit=TOP_K, expr="label == 1")
            total_filter_time += time.time() - start_query_filter

            # <<<<< Query - Range search (simulated as Milvus doesn't hae native Python API range search) >>>>>
            start_query_range = time.time()
            _ = collection.query(expr="label >= 0", output_fields=["id", "label"])
            total_range_time += time.time() - start_query_range

        recall = recall_at_k(milvus_results, gt_indices)
        avg_topk_time = total_topk_time / QUERY_COUNT
        avg_filter_time = total_filter_time / QUERY_COUNT
        avg_range_time = total_range_time / QUERY_COUNT

        # <<<<< Updates >>>>>
        update_ids = random.sample(range(size), min(50, size))
        print(f"Updating {len(update_ids)} on dataset size: {size}...")
        new_vec = np.random.rand(DIM).tolist()

        start_update = time.time()
        expr = f"id in {update_ids}"
        collection.delete(expr)
        update_labels = np.random.randint(0, 10, len(update_ids))
        collection.insert([update_ids, update_labels, [new_vec] * len(update_ids)])
        avg_update_time = (time.time() - start_update) / len(update_ids)

        # <<<<< Deletes >>>>>
        delete_ids = random.sample(range(size), min(50, size))
        print(f"Deleting {len(delete_ids)} on dataset size: {size}...")

        start_delete = time.time()
        expr = f"id in {delete_ids}"
        collection.delete(expr)
        avg_delete_time = (time.time() - start_delete) / len(delete_ids)

        # <<<<< Results >>>>>
        results.append({
            "DB": "Milvus",
            "Size": size,
            "Setup Time (s)": setup_time,
            "Insert Time (s)": insert_time,
            "Avg Update Time (s)": avg_update_time,
            "Avg Delete Time (s)": avg_delete_time,
            "Avg Query TopK Time (s)": avg_topk_time,
            "Avg Query Filter Time (s)": avg_filter_time,
            "Avg Query Range Time (s)": avg_range_time,
            f"Recall@{TOP_K}": recall
        })

        print(f"Milvus Benchmark - Size {size} done")
        print(f"Setup: {setup_time}s")
        print(f"Insert: {insert_time}s")
        print(f"Avg Update: {avg_update_time}s")
        print(f"Avg Delete: {avg_delete_time}s")
        print(f"Avg TopK: {avg_topk_time}s")
        print(f"Avg Filter: {avg_filter_time}s")
        print(f"Avg Range: {avg_range_time}s")
        print(f"Recall@{TOP_K}: {recall}")

    return results

# ------------------------------------
# Qdrant Benchmark
# ------------------------------------
def benchmark_qdrant():
    print("\n--- Benchmarking Qdrant ---")
    client = QdrantClient(host="127.0.0.1", port=6333)
    results = []

    for size in SIZES:
        print(f"\nTesting size = {size}...")

        # <<<<< Table creation >>>>>
        start_setup = time.time()
        if client.collection_exists("cifar"):
            client.delete_collection("cifar")
        client.create_collection(
            collection_name="cifar",
            vectors_config=VectorParams(size=DIM, distance=Distance.COSINE)
        )
        setup_time = time.time() - start_setup

        # Data
        sub_embeddings = embeddings[:size]
        sub_labels = np.random.randint(0, 10, size)
        ids = list(range(size))

        # <<<<< Inserts >>>>>
        print(f"Inserting with size: {size}...")
        BATCH_SIZE = 1000 # for inserting in smaller batches to avoid JSON payload error (32MB) by upserting all size points at once
        start_insert = time.time()
        for start_idx in range(0, size, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, size)
            client.upsert(
                collection_name="cifar",
                points=[
                    models.PointStruct(
                        id=int(i),
                        vector=sub_embeddings[i].tolist(),
                        payload={"label": int(sub_labels[i])}
                    )
                    for i in range(start_idx, end_idx)
                ]
            )
        insert_time = time.time() - start_insert

        gt_indices = compute_ground_truth(size)
        qdrant_results = []
        total_topk_time, total_filter_time, total_range_time = 0, 0, 0

        for qi in range(QUERY_COUNT):
            print(f"Querying functions - Loop {qi+1}")
            query_vec = queries[qi].tolist()

            # <<<<< Query - Top-K search >>>>>
            start_query_topk = time.time()
            topk_hits = client.query_points(
                collection_name="cifar",
                query=query_vec,
                limit=TOP_K
            )
            total_topk_time += time.time() - start_query_topk
            qdrant_results.append([point.id for point in topk_hits.points])

            # <<<<< Query - Top-K search with filter (label = 1) >>>>>
            start_query_filter = time.time()
            _ = client.query_points(
                collection_name="cifar",
                query=query_vec,
                limit=TOP_K,
                query_filter=models.Filter(
                    must=[models.FieldCondition(
                        key="label",
                        match=models.MatchValue(value=1)
                    )]
                )
            )
            total_filter_time += time.time() - start_query_filter

            # <<<<< Query - Range search (< 0.5 for L2) >>>>>
            start_query_range = time.time()
            _ = client.query_points(
                collection_name="cifar",
                query=query_vec,
                limit=size,
                score_threshold=0.5
            )
            total_range_time += time.time() - start_query_range

        recall = recall_at_k(qdrant_results, gt_indices)
        avg_topk_time = total_topk_time / QUERY_COUNT
        avg_filter_time = total_filter_time / QUERY_COUNT
        avg_range_time = total_range_time / QUERY_COUNT

        # <<<<< Updates >>>>>
        update_ids = random.sample(range(size), min(50, size))
        print(f"Updating {len(update_ids)} on dataset size: {size}...")
        new_vec = np.random.rand(DIM).tolist()

        start_update = time.time()
        client.delete(
            collection_name="cifar",
            points_selector=PointIdsList(points=[i for i in update_ids])
        )
        client.upsert(
            collection_name="cifar",
            points=[
                models.PointStruct(
                    id=int(i),
                    vector=new_vec,
                    payload={"label": int(np.random.randint(0, 10))}
                )
                for i in update_ids
            ]
        )
        avg_update_time = (time.time() - start_update) / len(update_ids)

        # <<<<< Deletes >>>>>
        delete_ids = random.sample(range(size), min(50, size))
        print(f"Deleting {len(delete_ids)} on dataset size: {size}...")

        start_delete = time.time()
        client.delete(
            collection_name="cifar",
            points_selector=PointIdsList(points=[i for i in delete_ids])
        )
        avg_delete_time = (time.time() - start_delete) / len(delete_ids)

        # <<<<< Results >>>>>
        results.append({
            "DB": "Qdrant",
            "Size": size,
            "Setup Time (s)": setup_time,
            "Insert Time (s)": insert_time,
            "Avg Update Time (s)": avg_update_time,
            "Avg Delete Time (s)": avg_delete_time,
            "Avg Query TopK Time (s)": avg_topk_time,
            "Avg Query Filter Time (s)": avg_filter_time,
            "Avg Query Range Time (s)": avg_range_time,
            f"Recall@{TOP_K}": recall
        })

        print(f"Qdrant Benchmark - Size {size} done")
        print(f"Setup: {setup_time}s")
        print(f"Insert: {insert_time}s")
        print(f"Avg Update: {avg_update_time}s")
        print(f"Avg Delete: {avg_delete_time}s")
        print(f"Avg TopK: {avg_topk_time}s")
        print(f"Avg Filter: {avg_filter_time}s")
        print(f"Avg Range: {avg_range_time}s")
        print(f"Recall@{TOP_K}: {recall}")
    
    return results

# ------------------------------------
# FAISS Benchmark
# ------------------------------------
def benchmark_faiss():
    print("\n--- Benchmarking FAISS ---")
    results = []

    for size in SIZES:
        print(f"\nTesting size = {size}...")

        # <<<<< Table creation (index creation) >>>>>
        start_setup = time.time()
        index = faiss.IndexFlatL2(DIM)
        setup_time = time.time() - start_setup

        # Data
        sub_embeddings = embeddings[:size]
        sub_labels = np.random.randint(0, 10, size)

        # <<<<< Inserts >>>>>
        print(f"Inserting with size: {size}...")
        start_insert = time.time()
        index.add(sub_embeddings)
        insert_time = time.time() - start_insert

        gt_indices = compute_ground_truth(size)
        faiss_results = []
        total_topk_time, total_filter_time, total_range_time = 0, 0, 0

        for qi in range(QUERY_COUNT):
            print(f"Querying functions - Loop {qi+1}")
            query_vec = queries[qi].reshape(1, -1)

            # <<<<< Query - Top-K search >>>>>
            start_query_topk = time.time()
            distances, indices = index.search(query_vec, TOP_K)
            total_topk_time += time.time() - start_query_topk
            faiss_results.append(list(indices[0]))

            # <<<<< Query - Top-K search with filter (label = 1) >>>>>
            start_query_filter = time.time()
            distances, indices = index.search(query_vec, size)
            filtered = [idx for idx in indices[0] if sub_labels[idx] == 1][:TOP_K]
            total_filter_time += time.time() - start_query_filter

            # <<<<< Query - Range search (distance < threshold) >>>>>
            start_query_range = time.time()
            distances, indices = index.search(query_vec, size)
            threshold = 1.0
            ranged = [idx for d, idx in zip(distances[0], indices[0]) if d < threshold]
            total_range_time += time.time() - start_query_range

        recall = recall_at_k(faiss_results, gt_indices)
        avg_topk_time = total_topk_time / QUERY_COUNT
        avg_filter_time = total_filter_time / QUERY_COUNT
        avg_range_time = total_range_time / QUERY_COUNT

        # <<<<< Updates >>>>>
        update_ids = random.sample(range(size), min(50, size))
        print(f"Updating {len(update_ids)} on dataset size: {size}...")
        new_vecs = np.random.rand(len(update_ids), DIM).astype(np.float32)
        temp_embeddings = sub_embeddings.copy()
        start_update = time.time()
        for idx, vec in zip(update_ids, new_vecs):
            temp_embeddings[idx] = vec
        temp_index = faiss.IndexFlatL2(DIM)
        temp_index.add(temp_embeddings)
        avg_update_time = (time.time() - start_update) / len(update_ids)

        # <<<<< Deletes >>>>>
        delete_ids = set(random.sample(range(size), min(50, size)))
        print(f"Deleting {len(delete_ids)} on dataset size: {size}...")
        start_delete = time.time()
        kept_embeddings = np.array(
            [v for i, v in enumerate(temp_embeddings) if i not in delete_ids],
            dtype=np.float32
        )
        if kept_embeddings.ndim == 1 and kept_embeddings.size > 0:
            kept_embeddings = kept_embeddings.reshape(1, -1)
        temp_index = faiss.IndexFlatL2(DIM)
        if kept_embeddings.shape[0] > 0:
            temp_index.add(kept_embeddings)
        avg_delete_time = (time.time() - start_delete) / max(len(delete_ids), 1)

        # <<<<< Results >>>>>
        results.append({
            "DB": "FAISS",
            "Size": size,
            "Setup Time (s)": setup_time,
            "Insert Time (s)": insert_time,
            "Avg Update Time (s)": avg_update_time,
            "Avg Delete Time (s)": avg_delete_time,
            "Avg Query TopK Time (s)": avg_topk_time,
            "Avg Query Filter Time (s)": avg_filter_time,
            "Avg Query Range Time (s)": avg_range_time,
            f"Recall@{TOP_K}": recall
        })

        print(f"FAISS Benchmark - Size {size} done")
        print(f"Setup: {setup_time}s")
        print(f"Insert: {insert_time}s")
        print(f"Avg Update: {avg_update_time}s")
        print(f"Avg Delete: {avg_delete_time}s")
        print(f"Avg TopK: {avg_topk_time}s")
        print(f"Avg Filter: {avg_filter_time}s")
        print(f"Avg Range: {avg_range_time}s")
        print(f"Recall@{TOP_K}: {recall}")

    return results

# ------------------------------------
# pgvector Benchmark
# ------------------------------------
def benchmark_pgvector():
    print("\n--- Benchmarking pgvector ---")
    engine = create_engine("postgresql+psycopg2://postgres:postgres@localhost:5432/postgres")
    results = []

    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        for size in SIZES:
            print(f"\nTesting size = {size}...")

            # <<<<< Table creation >>>>>
            start_setup = time.time()
            conn.execute(text("DROP TABLE IF EXISTS cifar"))
            conn.execute(text(f"""
                CREATE TABLE cifar (
                    id INT PRIMARY KEY,
                    label INT,
                    embedding vector({DIM})
                )
            """))
            setup_time = time.time() - start_setup

            # Data
            sub_embeddings = embeddings[:size]
            sub_labels = np.random.randint(0, 10, size)

            # <<<<< Inserts >>>>>
            print(f"Inserting with size: {size}...")
            BATCH_SIZE = 1000
            start_insert = time.time()

            with conn.connection.cursor() as cur:
                for start in range(0, size, BATCH_SIZE):
                    end = min(start + BATCH_SIZE, size)
                    rows = [
                        (int(i), int(sub_labels[i]), sub_embeddings[i].tolist())
                        for i in range(start, end)
                    ]
                    execute_values(
                        cur, "INSERT INTO cifar (id, label, embedding) VALUES %s", rows
                    )
            insert_time = time.time() - start_insert

            gt_indices = compute_ground_truth(size)
            pgvector_results = []
            total_topk_time, total_filter_time, total_range_time = 0, 0, 0

            for qi in range(QUERY_COUNT):
                print(f"Querying functions - Loop {qi+1}")
                query_vec = queries[qi].tolist()
                vec_str = "[" + ",".join(map(str, query_vec)) + "]"

                # <<<<< Query - Top-K search >>>>>
                start_query_topk = time.time()
                topk_hits = conn.execute(
                    text(f"SELECT id FROM cifar ORDER BY embedding <-> '{vec_str}' LIMIT {TOP_K}")
                ).fetchall()
                total_topk_time += time.time() - start_query_topk
                pgvector_results.append([hit[0] for hit in topk_hits])

                # <<<<< Query - Top-K search with filter (label = 1) >>>>>
                start_query_filter = time.time()
                _ = conn.execute(
                    text(f"SELECT id FROM cifar WHERE label = 1 ORDER BY embedding <-> '{vec_str}' LIMIT {TOP_K}")
                ).fetchall()
                total_filter_time += time.time() - start_query_filter

                # <<<<< Query - Range search (distance < threshold) >>>>>
                start_query_range = time.time()
                threshold = 1.0
                _ = conn.execute(
                    text(f"SELECT id FROM cifar WHERE embedding <-> '{vec_str}' < {threshold}")
                ).fetchall()
                total_range_time += time.time() - start_query_range

            recall = recall_at_k(pgvector_results, gt_indices)
            avg_topk_time = total_topk_time / QUERY_COUNT
            avg_filter_time = total_filter_time / QUERY_COUNT
            avg_range_time = total_range_time / QUERY_COUNT

            # <<<<< Updates >>>>>
            update_ids = random.sample(range(size), min(50, size))
            print(f"Updating {len(update_ids)} on dataset size: {size}...")
            start_update = time.time()
            for uid in update_ids:
                new_vec = np.random.rand(DIM).astype(np.float32)  # new random embedding
                new_vec_str = "[" + ",".join(map(str, new_vec.tolist())) + "]"
                conn.execute(text(f"UPDATE cifar SET embedding = '{new_vec_str}' WHERE id = {uid}"))
            avg_update_time = (time.time() - start_update) / len(update_ids)

            # <<<<< Deletes >>>>>
            delete_ids = random.sample(range(size), min(50, size))
            print(f"Deleting {len(delete_ids)} on dataset size: {size}...")
            start_delete = time.time()
            ids_str = ",".join(map(str, delete_ids))
            conn.execute(text(f"DELETE FROM cifar WHERE id IN ({ids_str})"))
            avg_delete_time = (time.time() - start_delete) / len(delete_ids)

            # <<<<< Results >>>>>
            results.append({
                "DB": "pgvector",
                "Size": size,
                "Setup Time (s)": setup_time,
                "Insert Time (s)": insert_time,
                "Avg Update Time (s)": avg_update_time,
                "Avg Delete Time (s)": avg_delete_time,
                "Avg Query TopK Time (s)": avg_topk_time,
                "Avg Query Filter Time (s)": avg_filter_time,
                "Avg Query Range Time (s)": avg_range_time,
                f"Recall@{TOP_K}": recall
            })

            print(f"pgvector Benchmark - Size {size} done")
            print(f"Setup: {setup_time}s")
            print(f"Insert: {insert_time}s")
            print(f"Avg Update: {avg_update_time}s")
            print(f"Avg Delete: {avg_delete_time}s")
            print(f"Avg TopK: {avg_topk_time}s")
            print(f"Avg Filter: {avg_filter_time}s")
            print(f"Avg Range: {avg_range_time}s")
            print(f"Recall@{TOP_K}: {recall}")

    return results

# ------------------------------------
# Run All Benchmarks
# ------------------------------------
all_results = []
all_results += benchmark_milvus()
all_results += benchmark_qdrant()
all_results += benchmark_faiss()
all_results += benchmark_pgvector()

df = pd.DataFrame(all_results)
df.to_csv("benchmark_results.csv", index=False)
