from typing import Any, Dict
from pyrfc import Connection


class SAPConnector:
    def __init__(self, conn_params: Dict[str, Any]):
        self.conn_params = conn_params

    def call(self, function_name: str, **kwargs) -> Dict[str, Any]:
        connection = Connection(**self.conn_params)
        return connection.call(function_name, **kwargs)

    def ping(self) -> bool:
        connection = Connection(**self.conn_params)
        info = connection.call("RFC_PING")
        return bool(info)
