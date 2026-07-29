"""Streamlit UI tests via Streamlit's own headless test harness.

Rendering the app exercises the config wiring, the store lookup and the sidebar, so a broken
import or a renamed setting fails here instead of in front of a reviewer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture(scope="module")
def app():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(APP, default_timeout=300).run()


def test_app_renders_without_exception(app):
    assert not app.exception, app.exception


def test_title_and_sidebar_present(app):
    assert any("Research Paper Answer Bot" in t.value for t in app.title)
    assert len(app.selectbox) >= 3  # embedding, chunker, strategy


def test_defaults_match_settings(app):
    from rag_bot.config import settings

    values = [box.value for box in app.selectbox]
    assert settings.embed_model in values
    assert settings.chunker in values
    assert settings.retrieval_strategy in values


def test_sample_questions_offered_on_first_load(app):
    assert len(app.button) >= 5


def test_chunk_count_metric_is_shown(app):
    assert any(metric.label == "Chunks indexed" for metric in app.metric)
