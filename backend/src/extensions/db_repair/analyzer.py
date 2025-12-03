class DatabaseAnalyzer:
    async def analyze(self, schema_data: dict) -> dict:
        issues = []

        for table, meta in schema_data.items():
            # Missing primary key
            if not any(c["constraint_type"] == "PRIMARY KEY" for c in meta["constraints"]):
                issues.append({
                    "table": table,
                    "type": "missing_primary_key",
                    "severity": "high",
                    "message": f"Table '{table}' has no PRIMARY KEY."
                })

            # Detect missing indexes on FK columns
            indexed_cols = [idx["indexdef"].split("(")[-1].split(")")[0] 
                            for idx in meta["indexes"]]

            for c in meta["constraints"]:
                if c["constraint_type"] == "FOREIGN KEY":
                    col = c["column_name"]
                    if col not in indexed_cols:
                        issues.append({
                            "table": table,
                            "type": "missing_fk_index",
                            "severity": "medium",
                            "message": f"Foreign key '{col}' lacks an index."
                        })

            # Detect dangerous types
            for col in meta["columns"]:
                if col["data_type"] in ("text", "jsonb"):
                    issues.append({
                        "table": table,
                        "type": "unbounded_column",
                        "severity": "low",
                        "message": f"Column '{col['column_name']}' uses unbounded type '{col['data_type']}'."
                })

        return {"issues": issues}


db_analyzer = DatabaseAnalyzer()
