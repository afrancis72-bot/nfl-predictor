"""Smoke-test the unified Streamlit landing page without live network access."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_command_center_renders():
    page=Path(__file__).resolve().parents[1]/"pages"/"0_Team_AI2_NFL_Command_Center.py"
    app=AppTest.from_file(str(page),default_timeout=20).run()
    assert not app.exception, [str(e.message) for e in app.exception]
    assert any("NFL Command Center" in str(x.value) for x in app.title)
    assert len(app.button)>=2
