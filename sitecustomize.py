from pathlib import Path


def _patch_streamlit_app() -> None:
    path = Path(__file__).resolve().parent / "asset_data_test_app" / "app.py"
    if not path.exists():
        return

    source = path.read_text(encoding="utf-8")

    old_options = '''                portfolio_options = ["均等配分"]
                if maximum_sharpe is not None:
                    portfolio_options.append("シャープレシオ最大")'''
    new