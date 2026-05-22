"""Gradio app: paste a GitHub repo URL, get a monthly commit forecast."""

from __future__ import annotations

import logging
import os

import gradio as gr
from dotenv import load_dotenv

from src.aggregate import MIN_MONTHS_FOR_FORECAST, to_monthly
from src.forecast import ChronosFineTuned, ChronosZeroShot, NaiveSeasonal
from src.github_fetch import RateLimitError, RepoNotFoundError, fetch_commits, parse_repo_url
from src.metrics import backtest
from src.plotting import make_figure

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME", "")
ADAPTER_ID = f"{HF_USERNAME}/chronos-github-commits" if HF_USERNAME else None
BASE_MODEL = "amazon/chronos-t5-small"

_ft = ChronosFineTuned(base_model_id=BASE_MODEL, adapter_id=ADAPTER_ID, allow_fallback=True)
_zs = ChronosZeroShot(model_id=BASE_MODEL)
_naive = NaiveSeasonal(period=12)


def predict(url: str, horizon: int):
    if not url or not url.strip():
        return None, "Please paste a GitHub repository URL.", None
    try:
        spec = parse_repo_url(url)
    except ValueError as e:
        return None, f"❌ {e}", None
    try:
        commits = fetch_commits(spec.owner, spec.repo, token=GITHUB_TOKEN)
    except RepoNotFoundError:
        return None, f"❌ {spec.owner}/{spec.repo} not found or private.", None
    except RateLimitError as e:
        return None, f"❌ GitHub rate limit hit. Reset at {e.reset_at.isoformat()}", None

    s = to_monthly(commits)
    if len(s) < 6:
        return None, f"❌ Not enough history ({len(s)} months) to forecast.", None

    warning = ""
    if len(s) < MIN_MONTHS_FOR_FORECAST:
        warning = (
            f"⚠️ Only {len(s)} months of history "
            f"(recommended ≥{MIN_MONTHS_FOR_FORECAST}). Forecast confidence is low.\n\n"
        )

    forecasts = [
        _ft.forecast(s, horizon),
        _zs.forecast(s, horizon),
        _naive.forecast(s, horizon),
    ]
    fig = make_figure(s, forecasts, repo_label=f"{spec.owner}/{spec.repo}")

    bt_df = None
    if len(s) >= max(12, horizon) + 6:
        bt_df = backtest(s, holdout=12, horizon=min(horizon, 12), forecasters=[_ft, _zs, _naive])
        bt_df = bt_df.round(2)

    label = (
        f"{warning}**{spec.owner}/{spec.repo}** · {len(s)} months of history · "
        f"forecast horizon {horizon} months"
    )
    return fig, label, bt_df


CSS = """
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 1100px !important;
}
#hero { text-align: center; padding: 32px 0 24px; }
#hero h1 { font-size: 2.4rem; margin: 0 0 8px; }
#hero p { color: #64748b; margin: 0; }
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), css=CSS, title="repo-pulse") as app:
        with gr.Column(elem_id="hero"):
            gr.Markdown("# Forecast any GitHub repo")
            gr.Markdown("Fine-tuned Chronos transformer · monthly horizon")

        with gr.Row():
            url = gr.Textbox(
                placeholder="github.com/pytorch/pytorch", label="Repository URL", scale=4
            )
            btn = gr.Button("Predict →", variant="primary", scale=1)
        horizon = gr.Slider(1, 12, value=6, step=1, label="Horizon (months)")
        gr.Markdown("Try: `pytorch/pytorch` · `facebook/react` · `rust-lang/rust`")

        with gr.Column(visible=False) as results:
            label = gr.Markdown()
            chart = gr.Plot()
            gr.Markdown("### Backtest on last 12 months (held-out)")
            table = gr.Dataframe(
                headers=["model", "smape", "mae", "latency_ms"],
                interactive=False,
            )

        def _predict_and_show(u, h):
            fig, lbl, bt = predict(u, h)
            return {
                results: gr.update(visible=fig is not None),
                chart: fig,
                label: lbl,
                table: bt,
            }

        btn.click(
            _predict_and_show,
            inputs=[url, horizon],
            outputs=[results, chart, label, table],
        )
        url.submit(
            _predict_and_show,
            inputs=[url, horizon],
            outputs=[results, chart, label, table],
        )

    return app


if __name__ == "__main__":
    build_app().launch()
