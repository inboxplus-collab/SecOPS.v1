import os

import asyncpg

from backend.src.extensions.db_repair.analyzer import db_analyzer
from backend.src.extensions.db_repair.patch_generator import patch_generator
from backend.src.extensions.db_repair.schema_introspector import schema_introspector


class DatabaseRepairAgent:
    async def run_full_check(self) -> dict:
        tables = await schema_introspector.fetch_tables()
        schema = {}

        for t in tables:
            schema[t] = {
                "columns": await schema_introspector.fetch_columns(t),
                "constraints": await schema_introspector.fetch_constraints(t),
                "indexes": await schema_introspector.fetch_indexes(t),
            }

        issues = await db_analyzer.analyze(schema)
        patches = patch_generator.build_patches(issues["issues"])

        return {
            "issues": issues["issues"],
            "patches": patches,
        }

    async def apply_patches(self, patches: list[str]):
        conn = await asyncpg.connect(os.getenv("DATABASE_URL"))
        for sql in patches:
            await conn.execute(sql)
        await conn.close()

        return {"status": "repaired", "patch_count": len(patches)}


db_repair_agent = DatabaseRepairAgent()
