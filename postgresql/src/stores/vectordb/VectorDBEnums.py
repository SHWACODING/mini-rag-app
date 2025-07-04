from enum import Enum

class VectorDBEnums(Enum):
    QDRANT = "QDRANT"
    PGVECTOR = "PGVECTOR"

class DistanceMethodEnums(Enum):
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    L1 = "l1"
    L2 = "l2"

class PgVectorTableSchemeEnums(Enum):
    ID = 'id'
    TEXT = 'text'
    VECTOR = 'vector'
    METADATA = 'metadata'
    CHUNK_ID = 'chunk_id'
    _PREFIX = 'pgvector'

class PgVectorDistanceMethodEnums(Enum):
    COSINE = 'vector_cosine_ops'
    DOT = 'vector_l2_ops'

class PgvectorIndexTypeEnums(Enum):
    HNSW = 'hnsw'
    IVFFLAT = 'ivfflat'


