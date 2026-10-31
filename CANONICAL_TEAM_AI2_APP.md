# Canonical Team AI² Pro — single production application

This repository's **feature/unified-ai2-platform** branch is the ONLY live
four-sport Team AI² Pro application the user operates.

- Unified Streamlit entry point: `unified_app.py`.
- PGA page shown in the user's screenshots: `ai2_pga_workspace.py`.
- PGA slate ranking and veto-review page: `ai2_pga_review_ui.py` and
  `ai2_pga_salary_review.py`.
- Other sports are linked through the same unified entry point.

**Do not make user-facing changes to** `afrancis72-bot/team-ai2-pro`,
`afrancis72-bot/pga-predictor`, `afrancis72-bot/NASCAR-predictor` or
the standalone NFL app. They are legacy or nonproduction code for this user.
Do not ask the user to switch apps or re-upload data to a different app.

Before any future release:
1. Check that the target feature exists in this branch and in the exact PGA UI.
2. Validate the applicable sport tests and a Streamlit screen smoke test.
3. Change this app only and verify the deployed Streamlit branch before asking
   the user to refresh/reboot.
4. Keep user approval/veto authority and existing lineups intact.
5. Do not confuse a GitHub merge with verified cloud deployment.

The name `nfl-predictor` is a historical repository name, NOT the app's scope.
A repository rename is optional and should occur only after verifying Streamlit
Cloud's repository link and setting up any needed redirects/reconfiguration.
Never delete/archive a repository or app without explicit user authorization and
verification that this production app is unaffected.
