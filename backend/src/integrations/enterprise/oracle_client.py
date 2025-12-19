import cx_Oracle
from typing import Dict, List


class OracleConnector:
    def __init__(self, dsn: str, user: str, password: str):
        self.dsn = dsn
        self.user = user
        self.password = password

    def query(self, sql: str) -> List[Dict]:
        connection = cx_Oracle.connect(self.user, self.password, self.dsn)
        cursor = connection.cursor()
        cursor.execute(sql)
        columns = [c[0] for c in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]

    def get_schema(self) -> List[Dict]:
        return self.query(
            "SELECT table_name, column_name, data_type FROM all_tab_columns"
        )
