import pytest

from lexiflow.asr.backends import ScriptedBackend
from lexiflow.config import LexiFlowConfig
from lexiflow.pipeline import LexiFlowPipeline

pytest.importorskip("textual")
pytest.importorskip("pytest_asyncio")

from lexiflow.ui.tui import LexiFlowTUI  # noqa: E402


def build_pipeline(tmp_path):
    config = LexiFlowConfig()
    config.state.database_path = tmp_path / "tui.db"
    config.asr.warmup = False
    return LexiFlowPipeline(config, backend=ScriptedBackend([], config.asr))


@pytest.mark.asyncio
async def test_tui_boots_and_renders_metrics(tmp_path):
    app = LexiFlowTUI(build_pipeline(tmp_path), refresh_seconds=10.0)
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one("#metrics")
        assert table.row_count == 8
        assert table.get_cell("utterances", app._metric_column) == "0"


@pytest.mark.asyncio
async def test_tui_demo_key_fills_the_transcript(tmp_path):
    app = LexiFlowTUI(build_pipeline(tmp_path), refresh_seconds=10.0)
    async with app.run_test() as pilot:
        await pilot.press("d")
        await pilot.pause()
        table = app.query_one("#metrics")
        assert int(table.get_cell("utterances", app._metric_column)) > 0
        assert app.query_one("#actions").children
        assert app._rendered_seq > 0


@pytest.mark.asyncio
async def test_tui_filter_limits_the_transcript(tmp_path):
    app = LexiFlowTUI(build_pipeline(tmp_path), refresh_seconds=10.0)
    async with app.run_test() as pilot:
        await pilot.press("d")
        await pilot.pause()
        await pilot.press("slash")
        await pilot.pause()
        assert app.query_one("#search").has_focus

        app.query_one("#search").value = "pricing"
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_text == "pricing"
        assert app.query_one("#transcript").has_focus

        await pilot.press("c")
        await pilot.pause()
        assert app.query_text == ""


@pytest.mark.asyncio
async def test_tui_export_writes_markdown(tmp_path):
    app = LexiFlowTUI(build_pipeline(tmp_path), refresh_seconds=10.0)
    async with app.run_test() as pilot:
        await pilot.press("d")
        await pilot.pause()
        await pilot.press("e")
        await pilot.pause()
        written = list(tmp_path.glob("*.md"))
        assert written and "## Action items" in written[0].read_text()
