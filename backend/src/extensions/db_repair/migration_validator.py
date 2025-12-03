import os
import subprocess


class MigrationValidator:
    def validate(self):
        result = subprocess.run(
            ["alembic", "check"],
            cwd=os.getcwd(),
            capture_output=True,
            text=True
        )

        return {
            "ok": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }


migration_validator = MigrationValidator()
