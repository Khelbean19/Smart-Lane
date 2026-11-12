import psycopg2
import os
from loguru import logger

POSTGRES_USERNAME = os.getenv("POSTGRES_USERNAME")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")

def main():
    conn_string = f"host='{POSTGRES_HOST}' " \
                  f"dbname='{POSTGRES_DB}' " \
                  f"user='{POSTGRES_USERNAME}' " \
                  f"password='{POSTGRES_PASSWORD}' " \
                  f"port={POSTGRES_PORT}"
    try:
        logger.info("Connecting to database...")
        connection = psycopg2.connect(conn_string)
        logger.info(f"Successfully connected to {POSTGRES_DB}")
        logger.info(f"PostgreSQL database running on {POSTGRES_HOST}:{POSTGRES_PORT}")
        return connection
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")