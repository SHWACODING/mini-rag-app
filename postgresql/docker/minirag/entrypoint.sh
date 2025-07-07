#!/bin/bash

set -e

echo "Running database migrations..."
cd /app/models/db_schemes/minirag/
alembic upgrade head
cd /app

# Start FastAPI (via CMD passed to container)
exec "$@"
