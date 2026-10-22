# TEAM AI² Pro — unified development prototype

**Not production-ready.** Existing NFL `app.py` is unchanged.

## Run
```bash
pip install -r requirements.txt
streamlit run unified_app.py
```

The existing four-sport logo is shown on the home screen. Each quadrant links to its sport (PGA top left, NFL top right, NASCAR bottom left, NBA bottom right). Navigation buttons provide a fallback.

## Persistent uploads
The new UI stores one current file per sport and category in SQLite. Uploading a replacement overwrites that sport/category's active file. Other sports/categories remain untouched. A saved file can be downloaded.

**Hosting warning:** SQLite is persistent only when the database directory is on a durable disk. Streamlit Community Cloud and many similar hosts have ephemeral local storage. Set `AI2_DATA_DIR` to a persistent mounted volume before using this feature with important slate files. For a multi-instance deployment, migrate to managed object storage and a database. No backups, upload history, or authentication are implemented yet.

## Current engine status
All four engine interfaces are **disabled**. This intentionally prevents the unified prototype from presenting experimental lineups as verified outputs. NFL Classic requires an upstream `role_confidence` feed before its experimental optimizer can work. Existing sport apps remain independent and are not yet migrated.

## Acceptance checks before release
1. Browser-test logo quadrants and navigation buttons on desktop/mobile.
2. Restart app and confirm saved uploads persist on the target hosting service.
3. Upload a replacement slate and confirm only the selected sport/category changes.
4. Validate salary/entry file schemas and record data freshness.
5. Connect each sport engine through a versioned adapter; do not import the full legacy Streamlit UI.
6. Run historical and failure-mode tests, including absent injury/role evidence.
7. Enable exports only when all validation gates pass.

## Deployment note
The Streamlit app entry point must be changed to `unified_app.py` **only after** these checks are completed. Keep the existing `app.py` production entry point for now.
