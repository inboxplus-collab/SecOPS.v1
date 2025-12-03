import pyodbc
from typing import Dict, List


class MSSQLConnector:
    def __init__(self, conn_str: str):
        self.conn_str = conn_str

    def query(self, sql: str) -> List[Dict]:
        conn = pyodbc.connect(self.conn_str)
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [c[0] for c in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]

    def get_schema(self) -> List[Dict]:
        return self.query(
            "SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS"
        )
