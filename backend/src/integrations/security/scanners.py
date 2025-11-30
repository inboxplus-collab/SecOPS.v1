from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any


@dataclass
class ScanFinding:
    id: str
    title: str
    severity: str
    description: str
    source: str
    metadata: Dict[str, Any]


class SecurityScannerClient:
    """
    Placeholder integration for container/image scanners (Trivy, Snyk, etc.)

    In a real deployment you can extend this to:
      - call `trivy` CLI in a container,
      - call Snyk API with SNYK_TOKEN, etc.
    """

    def __init__(self) -> None:
        self.trivy_enabled = bool(os.getenv("TRIVY_ENABLED"))
        self.snyk_token = os.getenv("SNYK_TOKEN")

    async def scan_image(self, image_ref: str) -> List[ScanFinding]:
        # MVP behavior: return empty unless enabled, to avoid unexpected noise.
        if not (self.trivy_enabled or self.snyk_token):
            return []

        # You can implement real scanner calls here later.
        return [
            ScanFinding(
                id=f"dummy-{image_ref}",
                title=f"Example finding for {image_ref}",
                severity="medium",
                description="This is a placeholder vulnerability. Wire to real scanner.",
                source="scanner",
                metadata={"image": image_ref, "detected_at": datetime.utcnow().isoformat()},
            )
        ]
