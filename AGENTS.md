# Agent development guardrails — production sports app

**THE ONLY ACTIVE SPORTS DEPLOYMENT** is Streamlit app `nfl-predictor` with:
- GitHub repository: `afrancis72-bot/nfl-predictor`
- Production deployment branch: `feature/unified-ai2-platform`
- Streamlit entry point: `unified_app.py`
- Sports: NFL, PGA, NASCAR and NBA, in this one application

This file is for automated coding assistants and maintainers. Do not use
`main/app.py`, `dev/showdown-game-script-v1/app.py`, the separate
`team-ai2-pro` repository or the retired PGA/NASCAR repositories as deployment
targets. They are **read-only historical references** for active sports work.

The trading scanner is a different application in
`afrancis72-bot/smallcap-momentum-lab`; **never** change that repository when
working on sports.

Before EACH write, verify the exact repository, branch and entry point. Supply
`branch="feature/unified-ai2-platform"` explicitly on every GitHub file
update. Never rely on a connector's default branch.

After EACH change: verify code at the live branch path; run automated regression
tests; confirm the Streamlit deployment is on this branch and displays the
expected build fingerprint. A successful commit alone is NOT evidence of a
successful deployment. No paid contest entries are approved automatically.

Keep user-approved portfolios untouched unless a new approval is provided.
Keep Classic QB vs opposing DST prohibited; Showdown may allow it.
