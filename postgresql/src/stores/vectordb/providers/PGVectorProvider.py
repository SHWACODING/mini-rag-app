from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnums, PgVectorTableSchemeEnums, PgVectorDistanceMethodEnums, PgvectorIndexTypeEnums
import logging
from typing import List
from models.db_schemes import RetrievedDocument
from sqlalchemy.sql import text as sql_text
import json

class PGVectorProvider(VectorDBInterface):
    def __init__(self, db_client, default_vector_size: int = 768, distance_method: str = None, index_threshold: int = 100):
        self.db_client = db_client
        self.default_vector_size = default_vector_size
        self.index_threshold = index_threshold
        
        if distance_method == DistanceMethodEnums.COSINE.value:
            distance_method = PgVectorDistanceMethodEnums.COSINE.value
        elif distance_method == DistanceMethodEnums.DOT_PRODUCT.value:
            distance_method = PgVectorDistanceMethodEnums.DOT.value
        
        self.distance_method = distance_method
        
        self.pgvector_table_prefix = PgVectorTableSchemeEnums._PREFIX.value
        
        self.logger = logging.getLogger("uvicorn")
        
        self.default_index_name = lambda collection_name: f"{collection_name}_vector_idx"
    
    async def connect(self):
        async with self.db_client() as session:
            try:
                # Check if vector extension already exists
                result = await session.execute(sql_text(
                    "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
                ))
                extension_exists = result.scalar_one_or_none()
                
                if not extension_exists:
                    # Only create if it doesn't exist
                    await session.execute(sql_text("CREATE EXTENSION vector;"))
                    await session.commit()
            except Exception as e:
                # If extension already exists or any other error, just log and continue
                self.logger.warning(f"Vector extension setup: {str(e)}")
                await session.rollback()
    
    async def disconnect(self):
        # PGVector does not require explicit disconnection like some other databases
        self.logger.info("Disconnected from PGVector database.")
        pass
    
    async def is_collection_existed(self, collection_name: str) -> bool:
        record = None
        
        async with self.db_client() as session:
            async with session.begin():
                list_tables = sql_text(
                    "SELECT * FROM pg_tables WHERE tablename = :collection_name"
                )
                results = await session.execute(list_tables, {"collection_name": collection_name})
                record = results.scalar_one_or_none()
        
        if record is None:
            self.logger.info("Collection {collection_name} does not exist.")
            return False
        return record
    
    async def list_all_collections(self) -> List[str]:
        collections = []
        
        async with self.db_client() as session:
            async with session.begin():
                list_tables = sql_text(
                    "SELECT tablename FROM pg_tables WHERE tablename LIKE :prefix"
                )
                results = await session.execute(list_tables, {"prefix": self.pgvector_table_prefix})
                collections = results.scalars().all()
        
        self.logger.info(f"List of all collections: {collections}")
        return collections

    async def get_collection_info(self, collection_name: str) -> dict:        
        async with self.db_client() as session:
            async with session.begin():
                table_info_sql = sql_text("""
                    SELECT schemaname, tablename, tableowner, tablespace, hasindexes
                    FROM pg_tables 
                    WHERE tablename = :collection_name
                """)
                
                count_sql = sql_text(f"SELECT COUNT(*) FROM {collection_name}")
                
                table_info = await session.execute(table_info_sql, {"collection_name": collection_name})
                record_count = await session.execute(count_sql)
                
                table_data = table_info.fetchone()
                if not table_data:
                    self.logger.info(f"Collection {collection_name} does not exist.")
                    return None
                
                return {
                    "table_info": {
                        "schemaname": table_data[0],
                        "tablename": table_data[1],
                        "tableowner": table_data[2],
                        "tablespace": table_data[3],
                        "hasindexes": table_data[4],
                    },
                    "record_count": record_count.scalar_one()
                }
    
    async def delete_collection(self, collection_name: str):
        if not await self.is_collection_existed(collection_name):
            self.logger.info(f"Collection {collection_name} does not exist, cannot delete.")
            return False
        
        async with self.db_client() as session:
            async with session.begin():
                self.logger.info(f"Deleting/Resetting PGVECTOR collection {collection_name}.")
                drop_table_sql = sql_text(f"DROP TABLE IF EXISTS {collection_name}")
                await session.execute(drop_table_sql)
                self.logger.info(f"Deleted collection: {collection_name}")
                await session.commit()
        
        return True
    
    async def create_collection (self, collection_name: str, embedding_size: int, do_reset: bool = False):
        if do_reset:
            _ = await self.delete_collection(collection_name=collection_name)
        
        is_collection_existed = await self.is_collection_existed(collection_name=collection_name)
        
        if not is_collection_existed:
            self.logger.info(f"Collection {collection_name} does not exist, creating it...")
            
            async with self.db_client() as session:
                async with session.begin():
                    create_table_sql = sql_text(
                        f'CREATE TABLE {collection_name} ('
                            f'{PgVectorTableSchemeEnums.ID.value} bigserial PRIMARY KEY, '
                            f'{PgVectorTableSchemeEnums.TEXT.value} text, '
                            f'{PgVectorTableSchemeEnums.VECTOR.value} vector({embedding_size}), '
                            f'{PgVectorTableSchemeEnums.METADATA.value} jsonb  DEFAULT \'{{}}\', '
                            f'{PgVectorTableSchemeEnums.CHUNK_ID.value} integer, '
                            f'FOREIGN KEY ({PgVectorTableSchemeEnums.CHUNK_ID.value}) REFERENCES data_chunks(chunk_id)'
                        ')'
                    )
                    
                    await session.execute(create_table_sql, {"collection_name": collection_name})
                    self.logger.info(f"Created collection: {collection_name} with embedding size: {embedding_size}")
                    await session.commit()
                    
            return True
        
        else:
            self.logger.info(f"Collection {collection_name} already exists, skipping creation.")
            return False
    
    async def is_index_existed(self, collection_name: str) -> bool:
        index_name = self.default_index_name(collection_name)
        
        async with self.db_client() as session:
            async with session.begin():
                check_sql = sql_text("""
                                    SELECT 1
                                    FROM pg_indexes
                                    WHERE tablename = :collection_name
                                    AND indexname = :index_name
                                    """)
                result = await session.execute(check_sql, {"collection_name": collection_name, "index_name": index_name})
                record = result.scalar_one_or_none()
        
        if record is None:
            self.logger.info(f"Index {index_name} does not exist for collection {collection_name}.")
            return False
        
        self.logger.info(f"Index {index_name} exists for collection {collection_name}.")
        return True
    
    async def create_vector_index(self, collection_name: str, index_type: str = PgvectorIndexTypeEnums.HNSW.value):
        is_index_existed = await self.is_index_existed(collection_name=collection_name)
        if is_index_existed:
            self.logger.info(f"Index for collection {collection_name} already exists, skipping creation.")
            return False
        
        if not await self.is_collection_existed(collection_name=collection_name):
            self.logger.error(f"Collection {collection_name} does not exist, cannot create index.")
            return False
        
        async with self.db_client() as session:
            async with session.begin():
                count_sql = sql_text(
                    f"SELECT COUNT(*) FROM {collection_name}"
                )
                count_result = await session.execute(count_sql)
                record_count = count_result.scalar_one()
                
                if record_count < self.index_threshold:
                    self.logger.info(f"Record count {record_count} is below threshold {self.index_threshold}, skipping index creation.")
                    return False
                
                self.logger.info(f"START :: Creating index for collection {collection_name} with type {index_type}.")
                
                index_name = self.default_index_name(collection_name)
                
                create_idx_sql = sql_text(
                    f"CREATE INDEX IF NOT EXISTS {index_name} ON {collection_name} "
                    f"USING {index_type} ({PgVectorTableSchemeEnums.VECTOR.value} {self.distance_method})"
                )
                
                await session.execute(create_idx_sql)
                
                self.logger.info(f"END :: Created index for collection {collection_name} with type {index_type}.")
                
    async def  reset_vector_index(self, collection_name: str, index_type: str = PgvectorIndexTypeEnums.HNSW.value) -> bool:
        index_name = self.default_index_name(collection_name)
        
        is_index_existed = await self.is_index_existed(collection_name=collection_name)
        if not is_index_existed:
            self.logger.info(f"Index {index_name} does not exist for collection {collection_name}, skipping reset.")
            return False
        
        if not await self.is_collection_existed(collection_name=collection_name):
            self.logger.error(f"Collection {collection_name} does not exist, cannot reset index.")
            return False
        
        async with self.db_client() as session:
            async with session.begin():
                drop_sql = sql_text(
                    f"DROP INDEX IF EXISTS {index_name}"
                )
                await session.execute(drop_sql)
                self.logger.info(f"Dropped index {index_name} for collection {collection_name}.")
        
        return await self.create_vector_index(collection_name=collection_name, index_type=index_type)
    
    async def insert_one(self, collection_name: str, text: str, vector: list, metadata: dict = None, record_id: str = None):
        is_collection_existed = await self.is_collection_existed(collection_name=collection_name)
        if not is_collection_existed:
            self.logger.error(f"Collection {collection_name} does not exist, cannot insert record.")
            return False
        
        if not record_id:
            self.logger.error("Record ID is required for insertion.")
            return False

        async with self.db_client() as session:
            async with session.begin():
                insert_sql = sql_text(
                    f"INSERT INTO {collection_name} "
                    f"({PgVectorTableSchemeEnums.TEXT.value}, "
                    f"{PgVectorTableSchemeEnums.VECTOR.value}, "
                    f"{PgVectorTableSchemeEnums.METADATA.value}), "
                    f"{PgVectorTableSchemeEnums.CHUNK_ID.value}) "
                    "VALUES (:text, :vector, :metadata, :chunk_id)"
                )
                
                metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else '{}'
                
                await session.execute(insert_sql, {
                    "text": text,
                    "vector": '[' + ",".join([ str(v) for v in vector ]) + ']',
                    "metadata": metadata_json,
                    "chunk_id": record_id
                })
                self.logger.info(f"Inserted record into {collection_name}: {record_id}")
                await session.commit()
                
                await self.create_vector_index(collection_name=collection_name)
        
        return True
        
    async def insert_many (self, collection_name: str, texts: List, vectors: List, metadata: List = None, record_ids: List = None, batch_size: int = 50):
        is_collection_existed = await self.is_collection_existed(collection_name=collection_name)
        if not is_collection_existed:
            self.logger.error(f"Collection {collection_name} does not exist, cannot insert record.")
            return False
        
        if not record_ids or len(record_ids) != len(vectors):
            self.logger.error("Record IDs are required for insertion and must match the number of vectors.")
            return False
        
        if not metadata or len(metadata) == 0:
            metadata = [None] * len(texts)
            
        
        async with self.db_client() as session:
            async with session.begin():
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_vectors = vectors[i:i + batch_size]
                    batch_metadata = metadata[i:i + batch_size]
                    batch_record_ids = record_ids[i:i + batch_size]
                    
                    values = []
                    
                    for _text, _vector, _metadata, _record_id in zip(batch_texts, batch_vectors, batch_metadata, batch_record_ids):
                        
                        metadata_json = json.dumps(_metadata, ensure_ascii=False) if _metadata else '{}'
                        
                        values.append({
                            "text": _text,
                            "vector": '[' + ",".join([ str(v) for v in _vector ]) + ']',
                            "metadata": metadata_json,
                            "chunk_id": _record_id
                        })
                    
                    batch_insert_sql = sql_text(
                        f"INSERT INTO {collection_name} "
                        f"({PgVectorTableSchemeEnums.TEXT.value}, "
                        f"{PgVectorTableSchemeEnums.VECTOR.value}, "
                        f"{PgVectorTableSchemeEnums.METADATA.value}, "
                        f"{PgVectorTableSchemeEnums.CHUNK_ID.value}) "
                        "VALUES (:text, :vector, :metadata, :chunk_id)"
                    )
                    await session.execute(batch_insert_sql, values)
                    self.logger.info(f"Inserted batch of records into {collection_name} from index {i} to {i + len(batch_texts) - 1}")
            
        await self.create_vector_index(collection_name=collection_name)
        return True
    
    async def search_by_vector (self, collection_name: str, vector: list, limit: int):
        is_collection_existed = await self.is_collection_existed(collection_name=collection_name)
        if not is_collection_existed:
            self.logger.error(f"Collection {collection_name} does not exist, cannot search..?")
            return False
        
        vector = '[' + ",".join([ str(v) for v in vector ]) + ']'
        
        async with self.db_client() as session:
            async with session.begin():
                search_sql = sql_text(
                    f"SELECT {PgVectorTableSchemeEnums.TEXT.value} as text, 1 - ({PgVectorTableSchemeEnums.VECTOR.value} <=> :vector) as score "
                    f"FROM {collection_name} "
                    "ORDER BY score DESC "
                    "LIMIT :limit"
                )
                
                results = await session.execute(search_sql, {"vector": vector, "limit": limit})
                records = results.fetchall()
                
                retrieved_docs = []
                for record in records:
                    retrieved_docs.append(RetrievedDocument(
                        text=record.text,
                        score=record.score,
                    ))
                
                self.logger.info(f"Retrieved {len(retrieved_docs)} documents from collection {collection_name}.")
                return retrieved_docs
