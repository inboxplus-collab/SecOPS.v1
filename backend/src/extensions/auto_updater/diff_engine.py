import difflib


class DiffEngine:

    def generate_diff(self, old: str, new: str, fname: str) -> str:
        diff = difflib.unified_diff(
            old.splitlines(),
            new.splitlines(),
            fromfile=f"{fname} (old)",
            tofile=f"{fname} (new)",
            lineterm=""
        )
        return "\n".join(diff)


diff_engine = DiffEngine()
