# jojocrypto

The jojo indicator for TradingView Pine Script v6 and Python. Both implementations
use the current v4-based, 14-component approximation on a 0 to 100 scale.

**jojo is a fitted approximation, not an exact reconstruction of the reference
indicator.** Its error varies by symbol and market conditions; see the measured
GRASSUSD limitation below.

## Files

| File | Purpose |
| --- | --- |
| [jojo.pine](jojo.pine) | TradingView indicator, rainbow plot, numeric states, markers, and alerts |
| [jojo.py](jojo.py) | Python calculation, component output, and CSV command-line interface |
| [requirements.txt](requirements.txt) | Pinned Python dependencies used for numerical verification |

This import preserves the existing formulas, coefficients, colors, thresholds,
and numeric state behavior. Display labels and state text are in English.

## TradingView

1. Open Pine Editor on a chart.
2. Paste the contents of `jojo.pine`, save, and add the indicator to the chart.
3. Use the indicator inputs to configure marker and state thresholds.

The main plot uses a high-contrast gradient from blue at 0 through cyan, green,
yellow, and orange to red at 100. Horizontal levels default to 80, 70, 50, and 20.
The top-right table shows the reading, persistent state, and bars in that state.

The first three numeric plots are `jojo`, `jojo state code`, and `Bars in state`.
The latter two are exposed in the Data Window for screening. State-change alerts
are available for all four nonzero states.

## Python

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Read a TradingView chart export or another OHLCV CSV:

```bash
python jojo.py chart_export.csv
python jojo.py chart_export.csv --tail 20
python jojo.py chart_export.csv --components -o result.csv
python jojo.py chart_export.csv --check jojo
```

Required columns are `open`, `high`, `low`, and `close`. `volume` and `time` are
optional, and names are case-insensitive, including TradingView's `Volume`.
The CLI interprets `time` as Unix seconds and displays dates in UTC.

Provide numeric data in chronological order, oldest bar first, for one symbol
and one timeframe. The script does not sort, fetch, resample, or adjust prices.
Include available earlier bars: recursive calculations depend on their history.
Without usable volume totals, the volume terms fall back to a neutral score of
50. Some price components may be undefined on flat or incomplete data.

Library usage:

```python
import pandas as pd
from jojo import compute_jojo

bars = pd.read_csv("chart_export.csv")
result = compute_jojo(bars, with_components=True)
print(result[["jojo", "state", "state_text", "buy", "sell"]].tail())

# Override the same thresholds used by the Pine inputs.
custom = compute_jojo(bars, ob_level=80.0, ob_exit=70.0)
```

The result preserves the input index and contains `jojo`, `state`, `state_text`,
`bars_in_state`, `state_changed`, `buy`, `sell`, and a hex `color`. Input `time`
is included when present. `with_components=True` adds all 14 `c_*` columns.

`--check COLUMN` reports numeric differences on rows where both values are
finite; it is a diagnostic, not a pass/fail test. It requires at least one such
row. Compare only against the same indicator version, inputs, price feed,
timeframe, and history. Older exports may contain a different formula under
the same `jojo` column name.

## States and defaults

| Code | State | Default entry or transition |
| ---: | --- | --- |
| 2 | Overbought | jojo reaches or exceeds 80 |
| 1 | Recovered from overbought | State 2 and jojo falls strictly below 70 |
| -1 | Recovered from oversold | State -2 and jojo rises strictly above 20 |
| -2 | Oversold | jojo reaches or falls below 20 |
| 0 | Unset | No extreme state has been entered yet |

These states are persistent. For example, state 2 remains active between 70 and
80 after an overbought entry. Recovery states remain active until another
extreme is entered. Extreme entries take precedence over recovery transitions.
`bars_in_state` starts at 1 on the first input bar, including warmup, and resets
to 1 on each transition.

| Python parameter | Default | Purpose |
| --- | ---: | --- |
| `ma_len` | 3 | SMA length for markers |
| `low_th` | 25 | Previous jojo must be below this for a buy marker |
| `high_th` | 85 | Previous jojo must be above this for a sell marker |
| `ob_level` | 80 | Overbought entry |
| `ob_exit` | 70 | Recovery from overbought |
| `os_level` | 20 | Oversold entry |
| `os_exit` | 20 | Recovery from oversold |

Buy markers require jojo to cross above its SMA; sell markers require a cross
below. The previous-bar threshold must also hold. Markers and state transitions
are separate outputs: a sell marker does not mean state 1 was entered.

## Model and validation limits

The components are ROC(10), MFI(14), CMF(21), PSY(14), Aroon up and inverted down,
Stoch RSI(7, 14), Bollinger z-score(22), stochastic K(17), normalized regression
slope(12), TSI(10, 9), Fisher(14), RCI(9), and RMI(2, 4). Their transformed values
are combined using fitted weights, with no final smoothing. Rounded weights
sum to 0.999998. Controlled probes informed some terms, but the complete
reference formula and its initialization remain unknown.

On the user-supplied COINBASE:GRASSUSD daily export examined on September 28,
2026, both indicators first produced a value on the 22nd bar. Of 34 input bars,
13 had comparable values:

| Comparison | Result |
| --- | ---: |
| Python vs exported jojo, maximum absolute numeric difference | 2.21e-13 |
| jojo vs reference, mean absolute error | 6.8919 points |
| jojo vs reference, maximum absolute error | 10.8833 points |
| First jojo reading at or above 80 | September 22, 2026 |
| First reference reading at or above 80 | September 25, 2026 |

The Python state codes, durations, and buy/sell markers also matched that jojo
export. This establishes parity for the checked sample, not universal
equivalence of every Pine built-in or agreement with the reference indicator.
Simple gain reductions improved GRASSUSD while worsening several earlier
symbols, so those experimental changes are not included here. Raw chart exports
are not distributed with this repository.

For this repository import, Python 3.12.12 with the pinned dependencies reproduced
the prior Python implementation exactly across 15 local exports / 7,387 input
bars, including component values, states, durations, markers, and colors. Only
display text changed. The Pine executable structure was compared with the
previous working version after excluding comments and strings; this import was
not recompiled in TradingView. Python compilation, CLI CSV output, and dependency
checks passed.

## Official references

- [TradingView Pine execution model](https://www.tradingview.com/pine-script-docs/language/execution-model/)
- [TradingView plots](https://www.tradingview.com/pine-script-docs/visuals/plots/)
- [pandas CSV reader](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)
- [NumPy documentation](https://numpy.org/doc/stable/)
- [pip requirements files](https://pip.pypa.io/en/stable/user_guide/#requirements-files)
