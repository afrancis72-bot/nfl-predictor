"""Runtime identity for the ONE intended unified sports deployment.

A code fingerprint is shown even when Streamlit's deployment lacks .git.
This identifies deployed CONTENT, not proof of build/release approval.
"""
import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path

DEPLOY_REPOSITORY = "afrancis72-bot/nfl-predictor"
DEPLOY_BRANCH = "feature/unified-ai2-platform"
DEPLOY_ENTRYPOINT = "unified_app.py"
RELEASE_LABEL = "2026-11-03 unified NFL portfolio correlation/audit repair"


@lru_cache(maxsize=1)
def deployed_build_identity():
    root = Path(__file__).resolve().parent
    paths = ("unified_app.py", "ai2_nfl_oneclick.py",
             "ai2_nfl_candidate_builder.py", "ai2_nfl_sim_portfolio.py",
             "ai2_nfl_correlation_guard.py", "ai2_nfl_portfolio_audit.py")
    digest = hashlib.sha256()
    for name in paths:
        path = root / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes() if path.exists() else b"<missing>")
    fingerprint = digest.hexdigest()[:16]
    commit = "not exposed by host"
    try:
        output = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=2,
                                check=False)
        if output.returncode == 0 and len(output.stdout.strip()) == 40:
            commit = output.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"repo": DEPLOY_REPOSITORY, "branch": DEPLOY_BRANCH,
            "entrypoint": DEPLOY_ENTRYPOINT, "release": RELEASE_LABEL,
            "fingerprint": fingerprint, "git_commit": commit}
