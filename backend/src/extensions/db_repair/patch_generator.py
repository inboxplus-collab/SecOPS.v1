class PatchGenerator:
    def generate_pk_patch(self, table: str):
        return f"""
        ALTER TABLE {table}
        ADD COLUMN id SERIAL PRIMARY KEY;
        """

    def generate_fk_index_patch(self, table: str, column: str):
        return f"""
        CREATE INDEX IF NOT EXISTS idx_{table}_{column}
        ON {table}({column});
        """

    def generate_safe_type_patch(self, table: str, column: str):
        return f"""
        ALTER TABLE {table}
        ALTER COLUMN {column} TYPE VARCHAR(255);
        """

    def build_patches(self, issues):
        patches = []
        for issue in issues:
            if issue["type"] == "missing_primary_key":
                patches.append(self.generate_pk_patch(issue["table"]))

            if issue["type"] == "missing_fk_index":
                patches.append(self.generate_fk_index_patch(
                    issue["table"], issue["message"].split("'")[1]
                ))

            if issue["type"] == "unbounded_column":
                patches.append(self.generate_safe_type_patch(
                    issue["table"],
                    issue["message"].split("'")[1]
                ))

        return patches


patch_generator = PatchGenerator()
