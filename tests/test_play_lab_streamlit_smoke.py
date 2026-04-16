import pytest


def test_streamlit_app_module_importable() -> None:
    pytest.importorskip("streamlit")
    import play_lab.streamlit_app as m

    assert callable(m.main)
