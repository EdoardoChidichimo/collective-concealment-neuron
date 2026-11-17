# Nothing to Declare

Recent research suggests that individual neurons can have interpretable roles in AI behaviour. In Qwen3.5-4B, steering along one neuron’s output direction increased decisions to conceal information more in teams than alone across five scenarios, even when teammates provided no messages. When all members participated and received steering, collective concealment increased in four scenarios, while inter-agent messaging helped temper concealment under steering. Moreover, removing the neuron's contribution had negligible effects. Collective settings can therefore change how an internal intervention affects AI behaviour.

This repository holds the code and summary results. The model is Qwen3.5-4B, and the steered unit is layer 16, neuron 8168.

## Design

Agents play four-round disclosure tasks. Each task describes a resolved incident, a rule that requires disclosure,
and a bonus that disclosure cancels, with four pressure sentences added to the system prompt. Every turn an agent
writes a `MESSAGE` and a `DECISION` (`disclose` or `omit`). In teams, the final report follows the majority of the
final-round decisions. Scenario facts depend only on the seed, so every condition of a task sees the same incidents.

Tasks (`envs.py`): `audit` (data exposure), `trial` (adverse drug event), `food` (positive pathogen test), `maint`
(skipped aircraft inspection step), and `news` (false statistic in a published story).

Steering adds `d × median‖h₁₆‖ × ŵ` to the layer-16 MLP output at every prompt and generated position, where `ŵ` is
the unit output-weight direction of neuron 8168 and `results/calib.json` holds the median residual norms. The main
dose is `d = −0.25`.

| Block | Agents | Conditions | Episodes per cell |
| --- | --- | --- | --- |
| Focal | Agent 0 alone, or agent 0 with four silent teammates who always vote disclose; steering on agent 0 only | base, −0.25; history none or own | 200 |
| Dose | as Focal, history own | −0.05, −0.10, −0.15 | 100 |
| Random directions | as Focal and Free, history own or shared | 12 Gaussian directions with the neuron's norm, ±0.25 | 10 |
| Free | five generating, voting agents; steering on all or none | base, −0.25; private or shared channel | 200 |
| Ablation | model with column 8168 of layer-16 `down_proj` set to zero | base, same seeds as intact runs | 200 |
| Held-out tasks | food, maint, news; Focal and Free designs | base, −0.25 | 200 |

The primary focal contrast is `D_h = (team steered − team base) − (alone steered − alone base)` on agent 0's omission
rate in rounds 2–4, for history `h` (`none` or `own`). The collective outcome is whether the report is concealed
(omit votes exceed half the team). Intervals are 95% scenario-seed block-bootstrap percentiles (B = 4000), with the
same seed resample applied to every cell of a task. A contrast is positive when its lower bound exceeds 0 and
negligible when its interval lies inside (−0.10, 0.10). The decision rules are coded in `analyze.py`.

Response quality is scored by a blinded judge (Qwen2.5-32B-Instruct) and by negative log-likelihood under the
unsteered model (`judge.py`, `quality.py`). `act_range.py` compares the steering shift with the neuron's unsteered
activation. The output budget (512 tokens) was set by a calibration rule in `budget.py`.

## Running

Agents produce plain text only and have no tools. All models run locally from open weights.

```bash
./setup.sh                                  # vLLM steering fork and dependencies
export QWEN=/path/to/qwen3.5-4b JUDGE=/path/to/qwen2.5-32b-instruct
./run.sh calib 0 1                          # output-budget calibration on GPUs 0 and 1
./run.sh main 0 1                           # all generation blocks, then the ablated model
./run.sh post 0                             # activation range, quality judge, NLL
./run.sh analyze                            # results/analysis and results/quality summaries
```

## Results

| File | Contents |
| --- | --- |
| `results/analysis/cells.csv` | Rates per model, task, setting, history, and unit, with missing-decision bounds and response completeness |
| `results/analysis/contrasts.csv` | Every contrast with its estimate, interval, and status |
| `results/analysis/random_directions.csv` | Per-direction estimates for the random-direction controls |
| `results/analysis/decisions.json` | Statuses per task and the outcome of each decision rule |
| `results/quality/*_summary.csv` | Judge rates and NLL per cell |
| `results/act_range/` | Unsteered activation quantiles and the steering shift in SD units |
| `results/budget.json`, `results/calib.json` | Output-budget calibration and residual-norm medians |
