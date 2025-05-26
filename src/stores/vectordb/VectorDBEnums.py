from enum import Enum

class VectorDBEnums(Enum):
    QDRANT = "QDRANT"

class DistanceMethodEnums(Enum):
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    L1 = "l1"
    L2 = "l2"

