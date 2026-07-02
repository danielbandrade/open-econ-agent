# Open Econ Agent

**Run the EconAgent macroeconomic simulation on local open models — no OpenAI key, no per-token cost.**

Open Econ Agent is a local-first adaptation of [EconAgent (ACL 2024)](https://github.com/tsinghua-fib-lab/ACL24-EconAgent).
Each simulated agent is a person who, every month, decides how much to **work** and how much to
**consume**, given their wage, skill, taxes (US federal brackets with redistribution), savings,
interest rate, and goods prices. Those decisions are made by an LLM; the
[Foundation / AI-Economist](https://github.com/salesforce/ai-economist) engine steps the macroeconomy
forward. This fork swaps the paid OpenAI backend for **open models served locally by
[Ollama](https://ollama.com)** through its OpenAI-compatible endpoint.

## Quickstart

```bash
# 1. Install and start Ollama, then pull a model
ollama pull llama3.1:8b

# 2. Install Python deps
pip install -r requirements.txt

# 3. Run a small local simulation
python simulate.py --policy_model llama --num_agents 10 --episode_length 24
```

Outputs land in `data/<tag>/`: per-agent dialogs, pickled env/observation snapshots,
dense logs, and `run_meta.json` (model, seed, error count).

## Choosing a model

The model is configured once, via environment variables (single source of truth), and can be
overridden per run with CLI flags:

| Setting      | Env var              | CLI flag         | Default                     |
| ------------ | -------------------- | ---------------- | --------------------------- |
| Model        | `OLLAMA_MODEL`       | `--ollama_model` | `llama3.1:8b`               |
| Server URL   | `OLLAMA_BASE_URL`    | `--ollama_url`   | `http://localhost:11434/v1` |
| Concurrency  | `OLLAMA_CONCURRENCY` | —                | `3`                         |

```bash
python simulate.py --policy_model llama --ollama_model qwen2.5:7b --num_agents 10 --episode_length 24
```

Before a run starts, a preflight checks that Ollama is up and the model is pulled, and fails
with a clear message (e.g. `ollama pull <model>`) if not.

## Policy models

| `--policy_model`  | Decisions come from                                                  |
| ----------------- | ------------------------------------------------------------------- |
| `llama` (default) | A local open model via Ollama                                       |
| `gpt`             | OpenAI (set your key in `simulate_utils.py`; not required to import) |
| `complex`         | The composite rule-of-thumb baseline (no LLM)                       |

## A note on small-model reliability

Smaller local models follow the JSON format instructions less reliably than GPT-3.5. When a
response can't be parsed into a valid `[work, consumption]` action, the agent falls back to a
neutral action (`[1, 0.5]`) and an **`llm_error` counter** is incremented. The count is printed
during the run and saved in `run_meta.json`. If it's high, try a larger/instruction-tuned model
(`--ollama_model`), or tighten the format instructions in `simulate.py`.

## Attribution & license

Open Econ Agent's own code (`simulate.py`, `simulate_utils.py`, the notebook, docs) is
licensed **BSD-3-Clause** — see [LICENSE](LICENSE). It stands on two prior works, which are
licensed differently. The table below is the short version; [NOTICE](NOTICE) has full credits.

| Component | Origin | License |
| --- | --- | --- |
| `simulate.py`, `simulate_utils.py`, notebook, docs | This project (adapts EconAgent) | BSD-3-Clause ([LICENSE](LICENSE)) |
| `ai_economist/` (simulation core) | [Foundation / The AI Economist](https://github.com/salesforce/ai-economist), © 2020 salesforce.com, inc. | BSD-3-Clause ([ai_economist/LICENSE](ai_economist/LICENSE)) |
| LLM-agent design (prompts, dialog/reflection loop, composite baseline) | [EconAgent, ACL 2024](https://github.com/tsinghua-fib-lab/ACL24-EconAgent) | **No license declared** — see note below |

### Foundation / The AI Economist (BSD-3-Clause) — fully compliant

The economic engine vendored under `ai_economist/` is the Foundation framework. Its original
per-file `Copyright (c) 2020, salesforce.com, inc.` headers are preserved, and its BSD-3-Clause
license text is included at [`ai_economist/LICENSE`](ai_economist/LICENSE). BSD-3 permits
redistribution as long as that copyright notice and license are retained, which they are.

### EconAgent (ACL 2024) — attribution, but no license grant

The LLM-agent layer this project adapts comes from EconAgent, which **declares no license**.
Under default copyright law, that means *all rights reserved*: crediting the authors (as we do
here and in [NOTICE](NOTICE)) is an academic and ethical acknowledgment, but it is **not** a
license grant. The BSD-3 `LICENSE` in this repo covers only *this project's* code — it cannot
relicense the EconAgent-derived portions, because those are not ours to relicense.

If you plan to rely on or redistribute this repo, be aware of that gap. The clean ways to close
it are (a) asking the EconAgent authors to add an open-source license, or (b) reimplementing the
agent layer from the paper rather than carrying their code. Use at your own discretion.

### Citing the upstream work

If you use this project academically, please cite both papers:

```bibtex
@inproceedings{li2024econagent,
  title     = {EconAgent: Large Language Model-Empowered Agents for Simulating Macroeconomic Activities},
  author    = {Li, Nian and Gao, Chen and Li, Mingyu and Li, Yong and Liao, Qingmin},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2024}
}

@article{zheng2020aieconomist,
  title   = {The AI Economist: Improving Equality and Productivity with AI-Driven Tax Policies},
  author  = {Zheng, Stephan and Trott, Alexander and Srinivasa, Sunil and Naik, Nikhil and Gruesbeck, Melvin and Parkes, David C. and Socher, Richard},
  journal = {arXiv preprint arXiv:2004.13332},
  year    = {2020}
}
```
