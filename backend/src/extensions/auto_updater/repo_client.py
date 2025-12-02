import base64
import httpx
import os


class RepoClient:
    def __init__(self):
        self.token = os.getenv("GITHUB_TOKEN")
        self.repo = os.getenv("TARGET_REPO")
        self.branch = os.getenv("TARGET_BRANCH", "main")

    async def get_file(self, path: str) -> tuple[str, str]:
        """
        Returns (file_content, sha)
        """
        url = f"https://api.github.com/repos/{self.repo}/contents/{path}"

        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers={
                "Authorization": f"Bearer {self.token}"
            })

        data = resp.json()
        content = base64.b64decode(data["content"]).decode()
        sha = data["sha"]

        return content, sha

    async def push_file(self, path: str, new_content: str, sha: str, message: str):
        payload = {
            "message": message,
            "content": base64.b64encode(new_content.encode()).decode(),
            "branch": self.branch,
            "sha": sha,
        }

        url = f"https://api.github.com/repos/{self.repo}/contents/{path}"

        async with httpx.AsyncClient() as client:
            await client.put(url, json=payload, headers={
                "Authorization": f"Bearer {self.token}"
            })


repo_client = RepoClient()
