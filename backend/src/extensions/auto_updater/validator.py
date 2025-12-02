import ast
import tempfile


class PatchValidator:

    def validate_python(self, new_code: str) -> bool:
        try:
            ast.parse(new_code)
        except Exception:
            return False
        return True


validator = PatchValidator()
