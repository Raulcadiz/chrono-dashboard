# ⏱️ Chrono Dashboard

Universal time-series event correlation explorer built on [chrono-correlator](https://github.com/Raulcadiz/chrono-correlator).

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41-red.svg)](https://streamlit.io)

## What it does

Pick a domain preset (or upload your own CSV/Excel) and get instant statistical
correlation analysis between a time-series metric and discrete events — no code required.

**Supported domains:** Personal Health · Public Health · Finance & Trading ·
Business & Sales · DevOps & Infra · Politics & Social Science ·
Science & Research · Energy & Environment · Digital Marketing · Custom

## Quick start

```bash
pip install -r requirements.txt
streamlit run chrono_dashboard.py
```

## Features

- 10 domain presets with realistic synthetic data
- CSV / Excel upload with automatic column detection
- Mann-Whitney U statistical analysis via chrono-correlator
- Lag sweep to find optimal pre-event window
- Bootstrap 95% confidence intervals
- AI narrative (Anthropic / Groq / Ollama) — optional
- Export: JSON · HTML · Markdown · CSV
- 5 tabs: Data · Visualization · Results · Narrative · Export

## AI narration (optional)

Set one of these environment variables to enable LLM-generated narratives:

```bash
export ANTHROPIC_API_KEY=your_key
# or
export GROQ_API_KEY=your_key
```

## License

AGPL-3.0 — free for personal and research use.
Corporations deploying this as a network service must open-source their modifications.
Commercial license available: g3ov3r@gmail.com

## Author

Raúl Gallardo · [GitHub @Raulcadiz](https://github.com/Raulcadiz)
