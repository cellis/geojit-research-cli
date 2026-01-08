#!/usr/bin/env python3
"""
Recreate the geojit-test database with fresh schema.
"""
import psycopg
import sys

DATABASE_NAME = "geojit-test"
POSTGRES_URL = "postgresql://localhost/postgres"


def recreate_database():
    """Drop and recreate the test database."""
    print(f"Recreating database '{DATABASE_NAME}'...")

    # Connect to postgres database to drop/create
    conn = psycopg.connect(POSTGRES_URL, autocommit=True)
    try:
        with conn.cursor() as cur:
            # Drop existing connections
            cur.execute(f"""
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = '{DATABASE_NAME}'
                  AND pid <> pg_backend_pid()
            """)

            # Drop database if exists
            cur.execute(f"DROP DATABASE IF EXISTS \"{DATABASE_NAME}\"")
            print(f"  ✓ Dropped existing database")

            # Create database
            cur.execute(f"CREATE DATABASE \"{DATABASE_NAME}\"")
            print(f"  ✓ Created database")
    finally:
        conn.close()

    # Now connect to the new database and set up schema
    from geojit.db import connect, ensure_schema

    db_url = f"postgresql://localhost/{DATABASE_NAME}"
    conn = connect(db_url)
    try:
        ensure_schema(conn)
        print(f"  ✓ Schema initialized")
    finally:
        conn.close()

    print(f"\nDatabase '{DATABASE_NAME}' is ready!")
    print(f"Connection string: {db_url}")


if __name__ == "__main__":
    try:
        recreate_database()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
