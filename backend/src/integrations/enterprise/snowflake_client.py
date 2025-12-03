import snowflake.connector
from typing import Dict, List


class SnowflakeConnector:
    def __init__(self, account: str, user: str, password: str, warehouse: str):
        self.account = account
        self.user = user
        self.password = password
        self.warehouse = warehouse

    def _connect(self):
        return snowflake.connector.connect(
            user=self.user,
            password=self.password,
            account=self.account,
            warehouse=self.warehouse,
        )

    def query(self, sql: str) -> List[Dict]:
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [c[0] for c in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]

    def get_schema(self, database: str, schema: str) -> List[Dict]:
        return self.query(
            f"SELECT table_name, column_name, data_type FROM {database}.information_schema.columns WHERE table_schema = '{schema}'"
        )
