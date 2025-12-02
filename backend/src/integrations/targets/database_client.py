from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from sqlalchemy import create_engine, inspect


logger = logging.getLogger(__name__)


@dataclass
class TableSummary:
    name: str
    columns: int
    notes: str


class DatabaseTargetClient:
    """
    Lightweight database target client.

    The client does not establish real network connections; instead it accepts
    declarative metadata about the target database to keep the orchestrator's
    contract simple and testable.
    """

    def __init__(self, connection_string: Optional[str] = None) -> None:
        self.connection_string = connection_string or ""

    def summarize(self, schema_metadata: Optional[dict] = None) -> Dict[str, object]:
        """
        Summarize a target database using provided metadata.

        It will attempt to introspect the database when a connection string is
        available, otherwise it uses the provided `schema_metadata` shape of:
        {"tables": [{"name": "users", "columns": 5, "notes": ""}]}.
        """
        tables = self._collect_table_summaries(schema_metadata)

        payload = {
            "connection": self.connection_string,
            "tables": [t.__dict__ for t in tables],
        }

        if schema_metadata is not None:
            payload["raw"] = json.dumps(schema_metadata)

        return payload

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _collect_table_summaries(self, schema_metadata: Optional[dict]) -> List[TableSummary]:
        if self.connection_string:
            tables = self._summarize_from_connection()
            if tables:
                return tables

        return self._summarize_from_metadata(schema_metadata)

    def _summarize_from_connection(self) -> List[TableSummary]:
        """Reflect tables from the configured database connection."""

        try:
            engine = create_engine(self.connection_string)
        except Exception as exc:  # pragma: no cover - best-effort logging
            logger.warning("DatabaseTargetClient: failed to build engine: %s", exc)
            return []

        try:
            inspector = inspect(engine)
            table_names = inspector.get_table_names()
            summaries: List[TableSummary] = []
            for name in table_names:
                try:
                    columns = inspector.get_columns(name)
                    comment = inspector.get_table_comment(name).get("text") or ""
                except Exception:
                    columns = []
                    comment = ""
                summaries.append(
                    TableSummary(
                        name=name,
                        columns=len(columns),
                        notes=comment,
                    )
                )
            return summaries
        except Exception as exc:  # pragma: no cover - best-effort logging
            logger.warning("DatabaseTargetClient: reflection failed: %s", exc)
            return []
        finally:
            try:
                engine.dispose()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                logger.debug("DatabaseTargetClient: engine dispose failed: %s", exc)

    def _summarize_from_metadata(self, schema_metadata: Optional[dict]) -> List[TableSummary]:
        tables_raw = schema_metadata.get("tables") if schema_metadata else []
        return [
            TableSummary(
                name=str(t.get("name", "unknown")),
                columns=int(t.get("columns", 0)),
                notes=t.get("notes", ""),
            )
            for t in tables_raw
        ]
