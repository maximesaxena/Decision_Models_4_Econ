# Jev4Econ
Testing whether out of the box "System 1" decision models could be useful for econ modelling. Code accompanies the pre-print "Decision Models as Scalable Economic Surrogates that Capture Human Variation"

We rerun experiments from two papers and one survey with decision models, and compare them with LLMs and with the human results:

| source | moniker | what humans did |
|---|---|---|
| Mei, Xie, Yuan & Jackson (2024, PNAS), "A Turing test of whether AI chatbots are behaviorally similar to humans" | `mei_et_al` | five classic economic games |
| Mas & Pallais (2017, AER), "Valuing Alternative Work Arrangements" | `mas_pallais` | chose between two call center jobs |
| New York Fed, SCE Household Spending Survey | `sce_spending` | said what they would do with 10% more, or less, income |

| model | kind | runs |
|---|---|---|
| `jev` | decision model | TypeSafe's API (`JEV_API_KEY`) |
| `jev-3rdparty` | decision model | the third-party provider jevmodel.org (`TYPESAFE_API_KEY`) |
| `kev` | decision model | a local Kev server ([jaredpalmer/kev](https://github.com/jaredpalmer/kev)) |
| `laya` | decision model | locally |
| `anyjev` | decision model | locally, on Qwen3.5-4B |
| `luna` | decision model | GPT-6 Luna, OpenAI's Decisions API (`OPENAI_API_KEY`) |
| `pplx` | decision model | pplx-decider-v1.1-27b, Perplexity's Decisions API (`PERPLEXITY_API_KEY`) |
| `mammoth` | decision model | Baseten's mammoth-1, Baseten's System One API (`BASETEN_API_KEY`) |
| `mercury`, `d1-ultra`, `microsoft`, `clef` | decision model | OpenRouter's Decisions API (`OPENROUTER_API_KEY`) |
| `qwen`, `qwen-nothink` | LLM | Qwen3.5-4B locally, with and without thinking |
| `gemini`, `gemini-lowthink` | LLM | Gemini 3.8 Flash, standard and lowest thinking (`GEMINI_API_KEY`) |
| `gemini-argon` | LLM | Gemini 4 Argon (`GEMINI_API_KEY`) |
| `gemma` | LLM | Gemma 4 26B through the Gemini API (`GEMMA_API_KEY`) |
| `haiku` | LLM | Claude Haiku 5.5, Anthropic's Messages API (`ANTHROPIC_API_KEY`) |
| `gpt`, `gpt-sol` | LLM | GPT-6 Luna and GPT-6 Sol as chat models, OpenAI's Responses API (`OPENAI_API_KEY`) |
| `glm-flash` | LLM | GLM-5.3-Flash, Perplexity's Agent API (`PERPLEXITY_API_KEY`) |

## Setup

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env             # then add your API keys
```

## Layout

```
src/common/models.py           the models, loaded by name; token limits; API prices
src/common/plotting.py         the figures' shared style, and GROUPS (which decision models --best tests)
src/experiments/<source>/      prompts, run, plotting/ and stat_tests/ scripts for each source
data/human/<source>/           human data (see each folder's README)
results/<source>/              every model call, written by the run scripts
analysis_results/<source>/     tables and charts, written by the plotting and stat_tests scripts
old/                           earlier scripts and results, not maintained
```

## How runs work

- Every model is asked each prompt `--repeats` times. Decision models answer with probabilities, which vary from call to call for Jev; LLMs are sampled.
- Every call's raw output is saved; all averaging happens in the analysis scripts.
- Calls are made one at a time and timed, excluding model loading and a warm-up call. Failed calls are retried, and only the attempt that worked is timed. Calls Laya retried internally are left out of the timings.
- Results are appended after every call, so an interrupted run resumes where it stopped.
- A call's cost is worked out from its tokens and `PRICES` in `models.py`, as no API reports one.
- The significance tests are bootstrap tests, Holm-corrected, at the 1% level. `--best` tests only the decision models in `GROUPS["Frontier DMs"]` in `src/common/plotting.py`.
- Plotting scripts with `--edit` open the figure with labels you can drag; pressing S saves their positions for later runs.

## Mei et al. (2024)

Five games (seven decisions): dictator, ultimatum (proposer, responder), trust (investor, banker), public goods and bomb risk. Decision models answer each game as a choice and as a score over the paper's 10 bins; LLMs get Mei et al.'s original prompt. Default: 50 repeats.

```
python src/experiments/mei_et_al/check_prompts.py
python src/experiments/mei_et_al/run.py jev laya luna gemini                    # every game
python src/experiments/mei_et_al/run.py jev --games dictator trust             # some games

python src/experiments/mei_et_al/plotting/compare_all_wasserstein.py [--edit]
python src/experiments/mei_et_al/plotting/compare_all_2.py [--edit]

python src/experiments/mei_et_al/stat_tests/bootstrap_tests.py [--best]
python src/experiments/mei_et_al/stat_tests/bootstrap_tests_kl.py [--best]
python src/experiments/mei_et_al/stat_tests/bootstrap_uniform.py
python src/experiments/mei_et_al/stat_tests/leave_one_out.py
```

| script | writes |
|---|---|
| `compare_all_wasserstein.py` | `mei.png`: Turing test win rate against Wasserstein-1 distance, and time against cost per call; `comparison.csv`, `summary_wasserstein.csv` |
| `compare_all_2.py` | `summary_2.png`: the same with KL divergence in place of Wasserstein-1; `comparison.csv`, `summary.csv` |
| `bootstrap_tests.py` | `bootstrap_tests.tex` (`_best` with `--best`): each decision model against each LLM, on Wasserstein-1 and the Turing test |
| `bootstrap_tests_kl.py` | `bootstrap_tests_kl.tex` (`_best`): the same on KL divergence and the Turing test |
| `bootstrap_uniform.py` | `bootstrap_uniform.tex`: the `--best` decision models against the uniform distribution |
| `leave_one_out.py` | `leave_one_out.tex`: `bootstrap_tests.py --best` rerun with each game left out |

## Mas & Pallais (2017)

Applicants chose between a standard job and an alternative arrangement. Models get every treatment × wage gap × maximum wage × order (580 prompts), and WTP is estimated for models and applicants with the paper's method. Default: 6 repeats.

```
python src/experiments/mas_pallais/run.py jev laya luna gemini
python src/experiments/mas_pallais/run.py gemini --pilot                       # LLMs: work from home at $16, once each

python src/experiments/mas_pallais/plotting/curves.py
python src/experiments/mas_pallais/plotting/final.py

python src/experiments/mas_pallais/stat_tests/bootstrap_tests.py [--best]
python src/experiments/mas_pallais/stat_tests/inattention.py
```

| script | writes |
|---|---|
| `curves.py` | `curves.png`: share choosing each job by wage gap, with fitted curves; `wtp.csv` |
| `final.py` | `mas.png`: work from home fitted curves, and the Wasserstein-1 distance of each model's WTP distribution from the applicants' |
| `bootstrap_tests.py` | `bootstrap_tests.tex` (`_best` with `--best`): each decision model against each LLM on that distance |
| `inattention.py` | `inattention.tex`: estimated inattention per treatment |

## SCE Household Spending Survey

1,000 respondents from 2020-24 are written up as personas from their demographics, and models answer both questions as each (2,000 prompts). Default: 1 repeat.

```
python src/experiments/sce_spending/data.py            # only to rebuild respondents.csv
python src/experiments/sce_spending/check_prompts.py
python src/experiments/sce_spending/run.py jev laya luna gemini
python src/experiments/sce_spending/run.py gemini --pilot                      # LLMs: first 100 personas, once each

python src/experiments/sce_spending/plotting/compare_all.py [--edit]

python src/experiments/sce_spending/stat_tests/bootstrap_tests.py [--best]
python src/experiments/sce_spending/stat_tests/bootstrap_uniform.py
```

| script | writes |
|---|---|
| `compare_all.py` | `turing_vs_fit.png`: Turing test win rate against KL divergence; `comparison.csv`, `summary.csv` |
| `bootstrap_tests.py` | `bootstrap_tests.tex` (`_best` with `--best`): paired tests of each decision model against each LLM |
| `bootstrap_uniform.py` | `bootstrap_uniform.tex`: the `--best` decision models against the uniform distribution |

### Robustness

The `--best` decision models and three LLMs rerun under one change to the prompts: `human` ("Act like a human." first, LLMs without a system prompt), `nosystem` (LLMs without a system prompt) or `wording` (QSP13new reworded to mirror QSP12n).

```
python src/experiments/sce_spending/robustness/run.py human                    # every model in the variant
python src/experiments/sce_spending/robustness/run.py nosystem gpt-sol --pilot
python src/experiments/sce_spending/robustness/analysis.py
```

`analysis.py` writes `robustness/`: `conditions.png` (each model's move from the baseline under each variant), `vs_baseline.tex`, `headline.tex` and `comparison.csv`.
