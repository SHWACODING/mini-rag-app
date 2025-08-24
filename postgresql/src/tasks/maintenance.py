from celery_app import celery_app, get_setup_utils
from helpers.config import get_settings
from utils.idempotency_manager import IdempotencyManager

import asyncio
import logging

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="tasks.maintenance.clean_celery_executions_table",
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 3, 'countdown': 60}
)
def clean_celery_executions_table(self):
    return asyncio.run(
        _clean_celery_executions_table(self)
    )


async def _clean_celery_executions_table(task_instance):
    
    settings = get_settings()
    
    try:
        
        (
            db_engine,
            db_client,
            llm_provider_factory,
            vectordb_provider_factory,
            generation_client,
            embedding_client,
            vector_db_client,
            template_parser
        ) = await get_setup_utils()
        
        # Create idempotency manager
        idempotency_manager = IdempotencyManager(db_client, db_engine)
        
        logger.warning(f"Cleaning !!!")
        _ = await idempotency_manager.cleanup_old_tasks(5)
        
        return True
    
    
    except Exception as e:
        logger.error(f"Task failed: {str(e)}")
        raise
    
    
    finally:
        try:
            if db_engine:
                await db_engine.dispose()
            
            if vector_db_client:
                await vector_db_client.disconnect()
        except Exception as e:
            logger.error(f"Task Failed While Cleaning: {str(e)}")
