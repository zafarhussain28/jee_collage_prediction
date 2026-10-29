from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_application_renders_without_errors():
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()

    assert not app.exception
    assert any("JEE College Predictor" in title.value for title in app.title)
    assert any(button.label == "Find matching programs" for button in app.button)


def test_valid_rank_submission_renders_real_results():
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30).run()
    app.number_input[0].set_value(10_000)
    app.button[0].click().run()

    assert not app.exception
    assert any(item.value == "Your results" for item in app.subheader)
    assert len(app.dataframe) == 1