from __future__ import annotations

import pytest

from core.checks.base import CheckContext
from core.checks.ci_hardening import CIHardeningCheck
from core.checks.k8s_misconfig import K8sMisconfigCheck


@pytest.mark.anyio
async def test_ci_hardening_check_runs_without_crash():
    check = CIHardeningCheck()
    ctx = CheckContext(extra={"github_repo": "owner/repo-does-not-exist"})
    # We don't care about results here, just that it doesn't raise
    try:
        await check.run(ctx)
    except Exception:
        pytest.fail("CIHardeningCheck.run raised unexpectedly")


@pytest.mark.anyio
async def test_k8s_misconfig_check_runs_without_crash():
    check = K8sMisconfigCheck()
    ctx = CheckContext()
    try:
        await check.run(ctx)
    except Exception:
        pytest.fail("K8sMisconfigCheck.run raised unexpectedly")
