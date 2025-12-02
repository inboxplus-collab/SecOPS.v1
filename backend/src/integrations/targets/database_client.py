from __future__ import annotations

from typing import Dict, List


class DatabaseClient:
    """Enterprise-aware database scanner.

    Each connector is optional and only runs when its driver and connection
    string are available via environment variables. The goal is to return
    actionable findings without failing the overall analysis run.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    def scan_databases(self) -> List[Dict]:
        findings: List[Dict] = []
        findings.extend(self._scan_postgres())
        findings.extend(self._scan_mysql())
        findings.extend(self._scan_mssql())
        findings.extend(self._scan_mongo())
        return findings

    # ------------------------------------------------------------------
    def _scan_postgres(self) -> List[Dict]:
        import os

        dsn = os.getenv("POSTGRES_DSN")
        if not dsn:
            return []

        findings: List[Dict] = []
        try:
            import psycopg2  # type: ignore
        except Exception:
            return []

        try:
            conn = psycopg2.connect(dsn, connect_timeout=5)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = 'public';
                """
            )
            for (name,) in cur.fetchall():
                findings.append(
                    {
                        "severity": "info",
                        "issue": f"Table '{name}' exists in public schema (consider restricted schemas).",
                        "source": "postgres",
                    }
                )
        except Exception:
            return findings
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return findings

    def _scan_mysql(self) -> List[Dict]:
        import os

        dsn = os.getenv("MYSQL_DSN")
        if not dsn:
            return []

        findings: List[Dict] = []
        try:
            import pymysql  # type: ignore
        except Exception:
            return []

        try:
            conn = pymysql.connect(dsn, connect_timeout=5)  # type: ignore[arg-type]
            cur = conn.cursor()
            cur.execute("SHOW DATABASES;")
            for (db_name,) in cur.fetchall():
                if db_name in {"information_schema", "mysql", "performance_schema"}:
                    continue
                findings.append(
                    {
                        "severity": "info",
                        "issue": f"Database '{db_name}' accessible; verify least privilege roles.",
                        "source": "mysql",
                    }
                )
        except Exception:
            return findings
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return findings

    def _scan_mssql(self) -> List[Dict]:
        import os

        dsn = os.getenv("MSSQL_DSN")
        if not dsn:
            return []

        findings: List[Dict] = []
        try:
            import pyodbc  # type: ignore
        except Exception:
            return []

        try:
            conn = pyodbc.connect(dsn, timeout=5)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sys.databases;")
            for (name,) in cur.fetchall():
                findings.append(
                    {
                        "severity": "info",
                        "issue": f"MSSQL database '{name}' detected; audit encryption and row-level security.",
                        "source": "mssql",
                    }
                )
        except Exception:
            return findings
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return findings

    def _scan_mongo(self) -> List[Dict]:
        import os

        uri = os.getenv("MONGO_URI")
        if not uri:
            return []

        findings: List[Dict] = []
        try:
            import pymongo  # type: ignore
        except Exception:
            return []

        try:
            client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=3000)
            db_names = client.list_database_names()
            for name in db_names:
                if name in {"admin", "local", "config"}:
                    continue
                findings.append(
                    {
                        "severity": "info",
                        "issue": f"Mongo database '{name}' reachable; ensure SCRAM auth and network policies.",
                        "source": "mongodb",
                    }
                )
        except Exception:
            return findings
        finally:
            try:
                client.close()
            except Exception:
                pass
        return findings
