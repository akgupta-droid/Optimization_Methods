# Prediction-Aware Robust Budget Allocation for Real-Time Bidding

This project combines calibrated click-through-rate prediction, clearing-price estimation and
hourly mixed-integer optimization to decide which advertising opportunities to pursue under a
budget cap. It is evaluated through a strict future replay for iPinYou advertisers **1458** and
**2997**.

The main question is deliberately narrow:

> Under the same campaign cap and auction stream, does allocating capacity using predicted click
> value outperform a policy that only maximizes impression volume?

The answer is **advertiser-dependent**. The CTR policy improves efficiency for advertiser 1458,
but produces little separation for advertiser 2997, whose CTR model has weak ranking power.

## Headline result

At the primary **50% train-derived budget cap**:

| Advertiser | Policy | Clicks | Volume clicks | Click lift | Spend (RMB) | Utilization | CPC improvement | Efficiency lift |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1458 | CTR central-cost | 203 | 195 | **+4.10%** | 13,654.59 | 30.3% | **+6.92%** | **+7.44%** |
| 1458 | CTR upper-cost | 246 | 195 | **+26.15%** | 17,103.19 | 38.0% | +3.79% | +3.94% |
| 2997 | CTR central-cost | 139 | 139 | 0.00% | 1,294.17 | 33.5% | +0.88% | +0.89% |
| 2997 | CTR upper-cost | 136 | 139 | **−2.16%** | 1,414.71 | 36.6% | **−10.74%** | **−9.70%** |

### What this means

- **The cleanest positive result is advertiser 1458 under the central-cost policy.** It obtains
  4.1% more clicks while spending about 3.1% less than the volume benchmark. This produces a
  7.44% lift in clicks per unit of spend.
- **Advertiser 2997 shows almost no benefit from CTR-based prioritization.** The central policy
  matches the volume benchmark's clicks with slightly lower spend, while the upper-cost policy is
  worse on clicks, CPC and efficiency.
- **The upper-cost result for 1458 should not be read as a pure robustness gain.** The upper cost is
  used both inside the budget constraint and as the submitted bid cap. It therefore bids more
  aggressively, spends 21.4% more than the volume benchmark and wins more auctions. Its 26.15%
  click lift becomes only a 3.94% efficiency lift after accounting for spend.
- **The scenario budget is not the active constraint in the primary replay.** Utilization ranges
  from 30% to 38%. Capacity, group quotas and auction-clearing conditions have more influence than
  the nominal cap, so click lift must be interpreted together with spend and win rate.

The results do not support a claim that robust or CTR-aware allocation universally wins. They show
one positive advertiser case, one largely neutral case and a clear limitation when the predictive
signal is weak.

## Why the advertisers behave differently

Clicks are extremely rare, so average precision relative to prevalence is more informative than
accuracy. AUC measures ranking, while calibration measures whether predicted probabilities have
the right scale.

| Diagnostic | Advertiser 1458 | Advertiser 2997 | Interpretation |
|---|---:|---:|---|
| Strict-test CTR prevalence | 0.0839% | 0.3707% | Clicks are rare for both advertisers |
| Strict-test ROC AUC | **0.620** | **0.543** | 1458 has modest ranking signal; 2997 is close to random |
| Strict-test average precision | 0.00157 | 0.00432 | Absolute AP reflects the different base rates |
| AP lift over prevalence | **1.87×** | **1.16×** | The 1458 model separates clicks more meaningfully |
| Platt calibration selected | Yes | Yes | Platt scaling performed best in chronological validation |
| ECE as a share of prevalence | 24.1% | 34.6% | Calibration error is material despite small absolute ECE |
| Predicted / actual test spend | 0.932 | 1.081 | Aggregate cost is underpredicted by 6.8% and overpredicted by 8.1% |
| Upper-cost test coverage | 77.4% | 82.9% | 1458 misses the 80% target; 2997 exceeds it |
| pCTR PSI | 0.011 | 0.022 | Limited score-distribution movement |
| Cost-prediction PSI | 0.026 | 0.120 | Cost distribution shifts more noticeably for 2997 |
| Strict-test duration | 3 days | 1 day | 2997 provides very limited independent temporal evidence |

### Advertiser 1458

The model is not exceptionally strong, but it has enough ranking signal to identify a more useful
subset of auctions than the volume objective. Test AUC falls from 0.642 in validation to 0.620,
and AP lift falls from 2.05× to 1.87×, so some degradation is present but the signal remains useful.

Aggregate cost prediction is reasonably close, although the model underestimates strict-test
spend by 6.8%. The upper bound covers 77.4% of test prices against an 80% target, which means the
calibrated uncertainty estimate does not transfer perfectly to the future period.

### Advertiser 2997

The model has little ranking power: test AUC is 0.543 and AP is only 1.16× prevalence. In that
setting, optimizing predicted clicks cannot reliably distinguish high-value opportunities from
the rest. This explains why the central policy effectively ties the volume benchmark and why a
more aggressive upper-cost bid does not improve performance.

The test cost distribution also moves more than it does for advertiser 1458, and the final holdout
contains only one strict day. The negative upper-cost result is real for this replay, but it is not
enough evidence to make a broader advertiser-level conclusion.

## Method

### 1. Data integrity and chronology

- Preserve the paper's `yyyyMMddHHmmssSSS` timestamp, including milliseconds.
- Consolidate repeated event records to one auction per Bid ID.
- Preserve any observed click with `max(click)` and reconcile repeated paying prices
  conservatively with `max(payprice)`.
- Remove all test rows that are not strictly later than the final training timestamp.
- Keep full SHA-256 hashes and field-level duplicate-resolution diagnostics.

The resulting strict test contains:

| Advertiser | Clean training rows | Training clicks | Strict-test rows | Strict-test clicks | Test days |
|---:|---:|---:|---:|---:|---:|
| 1458 | 3,077,016 | 2,454 | 613,990 | 515 | 3 |
| 2997 | 311,446 | 1,386 | 110,857 | 411 | 1 |

Advertiser 2997 loses 45,150 overlapping test records under the strict chronology rule.

### 2. Chronological model selection

Training is divided into four ordered roles:

1. **Fit:** LightGBM training and an internal chronological parameter search.
2. **Calibration:** Platt/isotonic probability calibration and one-sided cost calibration.
3. **Validation:** calibration selection and development diagnostics.
4. **Strict future test:** one final replay; never used for tuning.

The search evaluates eight CTR configurations and six clearing-price configurations, including the
original baseline. Early stopping chooses the required number of boosting rounds.

### 3. Prediction models

- **CTR model:** LightGBM classifier with Platt or isotonic calibration.
- **Central cost model:** LightGBM regressor for expected clearing price.
- **Upper cost model:** quantile LightGBM plus a split-conformal adjustment targeting 80% marginal
  coverage.

Identifiers, logged bid price, paying price, click and other post-auction fields are excluded from
the feature set.

### 4. Hourly allocation

For advertiser, hour and frozen planning group (g), the central policy solves:

\[
\max_x \sum_g \hat p_g x_g
\]

subject to:

\[
\sum_g \hat c_g x_g \le B_h,
\qquad 0 \le x_g \le N_g,
\qquad x_g \in \mathbb Z_+.
\]

Here, \(\hat p_g\) is calibrated expected CTR, \(\hat c_g\) is predicted cost, \(N_g\) is
forecast capacity and \(B_h\) is the adaptively paced hourly budget.

The **upper-cost policy** replaces \(\hat c_g\) with \(u_g \ge \hat c_g\). This is the box-robust
counterpart for the declared group-level cost interval. The **volume benchmark** replaces the CTR
objective with one while retaining the central-cost constraints.

### 5. Bid-aware future replay

The replay follows the evaluation logic in Zhang et al.:

1. process opportunities in timestamp order;
2. allocate a group quota before observing auction outcomes;
3. submit the policy's row-level cost estimate as the bid cap;
4. win only if the bid clears both `payprice` and `slotprice`;
5. charge the logged paying price;
6. count a click only for a won impression;
7. reoptimize the next hour using the remaining realized budget.

Unlike the paper, this implementation permits no last-auction overspend.

## Paper alignment and replay boundary

The methodological anchor is:

> Zhang, Yuan, Wang and Shen, [*Real-Time Bidding Benchmarking with iPinYou Dataset*](https://arxiv.org/abs/1407.7073).

| Paper protocol | Implementation here |
|---|---|
| Bid ID joins event logs | Repeated event rows are consolidated to one auction |
| Millisecond timestamps | All 17 timestamp digits are retained |
| Ascending-time evaluation | Strict test is processed sequentially |
| Bid clears market price and floor | Both `payprice` and `slotprice` are checked |
| Win is charged paying price | Realized spend decreases by `payprice` |
| Budgets of 1/32, 1/8 and 1/2 of test cost | Retained as a separate benchmark sensitivity |

The available files match the paper's **impression-log counts**, not its complete bid-request
counts. The experiment can evaluate lower or selective bids on historically observed impressions,
but cannot recover outcomes for auctions the historical policy lost. It is therefore a
**supported-impression replay**, not a complete reproduction of the paper's bid-log simulator.

The headline budget is derived from training-period capacity and predicted cost. The paper-style
test-cost budgets are reported separately because they use future realized information.

## Correctness checks

- MILP solutions match exhaustive enumeration on deterministic small instances.
- All hourly solves report optimal status.
- Maximum reported MIP gaps are below the declared tolerance.
- No replay exceeds its campaign budget.
- Every accepted auction clears both market price and slot floor.
- Fit, calibration, validation and strict test remain temporally ordered.
- Data hashes and duplicate-resolution decisions are preserved.

## Notebook structure

| Notebook | Purpose | Main output |
|---|---|---|
| `01_data_integrity_and_splits.ipynb` | Data audit, Bid-ID consolidation and chronological splits | Prepared Parquet files and hashes |
| `02_model_tuning_and_calibration.ipynb` | CTR/cost tuning, calibration and frozen planning groups | Models, diagnostics and scored test |
| `03_allocation_and_replay.ipynb` | Hourly MILPs and sequential auction replay | Policy results and replay traces |
| `04_results_checks_and_claims.ipynb` | Solver tests, diagnostics, plots and result wording | Final evidence tables |

Run the notebooks in order from a cleared kernel. Each stage writes an explicit artifact contract
for the next stage instead of relying on hidden notebook state.

## Running the project

Set the directory containing the advertiser folders:

```bash
export IPINYOU_DATA_ROOT=/path/to/ipinyou
```

Expected layout:

```text
<IPINYOU_DATA_ROOT>/
├── 1458/
│   ├── train.log.txt
│   └── test.log.txt
└── 2997/
    ├── train.log.txt
    └── test.log.txt
```

For a fast structural run:

```bash
export IPINYOU_SMOKE_TEST=1
```

Dependencies:

```text
numpy
pandas
pyarrow
matplotlib
scikit-learn
lightgbm
scipy
joblib
```

## Limitations

1. The replay observes only auctions present in the released impression stream.
2. Logged market prices are assumed not to change under the evaluated policy.
3. The study contains only two advertisers and one to three strict-test days.
4. CTR predicts response, not incremental causal advertising lift.
5. The budget caps are experimental train-derived scenarios, not contractual campaign budgets.
6. Low primary utilization means the budget is frequently non-binding.
7. Row-level upper-cost coverage is not a campaign-level probabilistic guarantee.
8. The upper-cost policy changes both allocation cost and bid cap; it does not isolate the effect
   of robust optimization from more aggressive bidding.
9. The capped hyperparameter search is not exhaustive.

## Recommended next experiment

Decouple the planning coefficient from the submitted bid:

- retain upper cost inside the MILP budget constraint;
- use the same central or value-based bid function for central and upper-cost policies;
- compare policies at matched realized spend;
- add more advertisers and independent future weeks.

That experiment would isolate whether robust budget accounting improves allocation, rather than
combining it with a different auction-winning rule.

## Claims supported by this project

- CTR-aware allocation improved advertiser 1458's replay efficiency under the central-cost rule.
- The same policy did not increase clicks for advertiser 2997.
- Cost uncertainty and predictive value differ materially by advertiser.
- All reported MILP solutions and budget transitions passed the implemented correctness checks.

## Claims not supported

- universal superiority over volume allocation;
- causal or incremental advertising lift;
- statistical significance across advertisers;
- an optimal production bidding policy;
- guaranteed campaign-level spend coverage.

