import asyncpg
import os


class SchemaIntrospector:
    async def connect(self):
        return await asyncpg.connect(os.getenv("DATABASE_URL"))

    async def fetch_tables(self):
        conn = await self.connect()
        q = """
        SELECT table_name 
        FROM information_schema.tables
        WHERE table_schema = 'public';
        """
        rows = await conn.fetch(q)
        await conn.close()
        return [r["table_name"] for r in rows]

    async def fetch_columns(self, table):
        conn = await self.connect()
        q = f"""
        SELECT column_name, data_type, is_nullable 
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = '{table}';
        """
        rows = await conn.fetch(q)
        await conn.close()
        return rows

    async def fetch_constraints(self, table):
        conn = await self.connect()
        q = f"""
        SELECT tc.constraint_type, kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
        ON tc.constraint_name = kcu.constraint_name
        WHERE tc.table_name = '{table}';
        """
        rows = await conn.fetch(q)
        await conn.close()
        return rows

    async def fetch_indexes(self, table):
        conn = await self.connect()
        q = f"""
        SELECT indexname, indexdef 
        FROM pg_indexes 
        WHERE schemaname = 'public' 
        AND tablename = '{table}';
        """
        rows = await conn.fetch(q)
        await conn.close()
        return rows


schema_introspector = SchemaIntrospector()
