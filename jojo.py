"""jojo indicator — Python port of jojo.pine (TradingView Pine Script v6).

This is the current v4-based approximation, not the reference indicator's source.
Its numeric output matches the checked GRASSUSD TradingView export within 2.21e-13.
That parity does not establish universal equivalence of TradingView built-ins or
an exact match to the reference indicator. See README.md for known limitations.

Implementation choices retained from the validated local version:
  * EMA and RMA seed from an initial full-window SMA.
  * Standard deviation uses the population definition.
  * Highest/lowest offset ties select the oldest value in the full window.
  * Rolling helpers use full windows and propagate missing values.

Usage as a library:
    import pandas as pd
    from jojo import compute_jojo
    out = compute_jojo(df)          # df has columns open, high, low, close, volume

Usage from the command line (TradingView "Export chart data" CSV works as-is):
    python3 jojo.py BATS_QCOM_1D.csv                 # print the last 10 bars
    python3 jojo.py BATS_QCOM_1D.csv -o result.csv   # write all bars to a CSV
    python3 jojo.py export.csv --check jojo          # compare with a jojo column exported from TradingView
"""
import argparse
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- default inputs (same as jojo.pine)
DEFAULTS = dict(ma_len=3, low_th=25.0, high_th=85.0,
                ob_level=80.0, ob_exit=70.0, os_level=20.0, os_exit=20.0)

# Weights: 3 probe-derived terms and 11 fitted oscillators; rounded sum is 0.999998.
W_ROC, W_MFI, W_CMF = 0.043770, 0.047316, 0.048732
W_FIT = {'psy': 0.05025, 'aroon_up': 0.06742, 'aroon_dn': 0.06397, 'srsi': 0.04256, 'bb': 0.10131,
         'k17': 0.06222, 'slope': 0.06667, 'tsi': 0.19263, 'fish': 0.11529, 'rci': 0.05961, 'rmi': 0.03825}

STATE_TEXT = {2: 'Overbought', 1: 'Recovered from overbought',
              -1: 'Recovered from oversold', -2: 'Oversold', 0: 'Unset'}


# ---------------------------------------------------------------- TradingView built-ins
def _shift(x, k=1):
    out = np.full(len(x), np.nan)
    if k > 0:
        out[k:] = x[:-k]
    else:
        out[:] = x
    return out


def _change(x, k=1):
    return x - _shift(x, k)


def _nz(x, v=0.0):
    x = x.copy()
    x[np.isnan(x)] = v
    return x


def _window_apply(x, n, fn):
    """Apply fn to every full window of length n; na if the window contains na."""
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        w = x[i - n + 1:i + 1]
        if not np.any(np.isnan(w)):
            out[i] = fn(w)
    return out


def _sum(x, n):     # math.sum
    return _window_apply(x, n, np.sum)


def _sma(x, n):     # ta.sma
    return _window_apply(x, n, np.mean)


def _stdev(x, n):   # ta.stdev(biased=true)
    return _window_apply(x, n, lambda w: np.sqrt(np.mean((w - w.mean()) ** 2)))


def _highest(x, n):
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        out[i] = np.max(x[i - n + 1:i + 1])
    return out


def _lowest(x, n):
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        out[i] = np.min(x[i - n + 1:i + 1])
    return out


def _highestbars(x, n):
    return _window_apply(x, n, lambda w: -(n - 1 - int(np.argmax(w))))


def _lowestbars(x, n):
    return _window_apply(x, n, lambda w: -(n - 1 - int(np.argmin(w))))


def _recursive(x, alpha, seed):
    out = np.full(len(x), np.nan)
    prev = np.nan
    for i in range(len(x)):
        if np.isnan(prev):
            cur = seed[i]
        else:
            cur = alpha * x[i] + (1 - alpha) * prev if not np.isnan(x[i]) else prev
        out[i] = cur
        prev = cur
    return out


def _rma(x, n):     # ta.rma
    return _recursive(x, 1.0 / n, _sma(x, n))


def _ema(x, n):     # ta.ema
    return _recursive(x, 2.0 / (n + 1), _sma(x, n))


def _rsi(x, n):     # ta.rsi
    d = _change(x)
    up = np.where(np.isnan(d), np.nan, np.maximum(d, 0))
    dn = np.where(np.isnan(d), np.nan, np.maximum(-d, 0))
    ru, rd = _rma(up, n), _rma(dn, n)
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.where(rd == 0, 100.0, np.where(ru == 0, 0.0, 100 - 100 / (1 + ru / rd)))
    res[np.isnan(ru) | np.isnan(rd)] = np.nan
    return res


def _stoch(src, high, low, n):   # ta.stoch
    hh, ll = _highest(high, n), _lowest(low, n)
    with np.errstate(divide='ignore', invalid='ignore'):
        return 100 * (src - ll) / (hh - ll)


def _tsi(x, short, long):        # ta.tsi (returns -1..1)
    pc = _change(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        return _ema(_ema(pc, long), short) / _ema(_ema(np.abs(pc), long), short)


def _clamp100(x):
    return np.where(np.isnan(x), np.nan, np.minimum(100.0, np.maximum(0.0, x)))


# ---------------------------------------------------------------- custom functions from jojo.pine
def _slope_z(src, n):
    """Least-squares slope over the last n bars divided by the population stdev (f_slopeZ)."""
    x = np.arange(n) - (n - 1) / 2.0
    ym, sd = _sma(src, n), _stdev(src, n)
    out = np.full(len(src), np.nan)
    for i in range(n - 1, len(src)):
        if sd[i] > 0:
            out[i] = (x @ (src[i - n + 1:i + 1] - ym[i])) / (x @ x) / sd[i]
    return out


def _fisher(high, low, n):
    hl2 = (high + low) / 2
    hh, ll = _highest(hl2, n), _lowest(hl2, n)
    val = np.full(len(hl2), np.nan)
    fish = np.full(len(hl2), np.nan)
    for i in range(len(hl2)):
        if np.isnan(hh[i]):
            continue
        vp = 0.0 if i == 0 or np.isnan(val[i - 1]) else val[i - 1]
        fp = 0.0 if i == 0 or np.isnan(fish[i - 1]) else fish[i - 1]
        v = 0.66 * ((hl2[i] - ll[i]) / max(hh[i] - ll[i], 1e-12) - 0.5) + 0.67 * vp
        val[i] = max(-0.999, min(0.999, v))
        fish[i] = 0.5 * np.log((1 + val[i]) / (1 - val[i])) + 0.5 * fp
    return fish


def _rci(src, n):
    """Rank correlation index; price ties rank the newer bar first (f_rci)."""
    out = np.full(len(src), np.nan)
    for i in range(n - 1, len(src)):
        w = src[i - n + 1:i + 1][::-1]                 # w[0] = current bar, w[k] = k bars ago
        if np.any(np.isnan(w)):
            continue
        order = np.argsort(-w, kind='stable')
        rank = np.empty(n)
        rank[order] = np.arange(1, n + 1)
        d = np.arange(1, n + 1) - rank
        out[i] = (1 - 6 * np.sum(d * d) / (n * (n * n - 1))) * 100
    return out


def _gradient(v, lo, hi, c_lo, c_hi):
    t = min(1.0, max(0.0, (v - lo) / (hi - lo)))
    a = [int(c_lo[k:k + 2], 16) for k in (1, 3, 5)]
    b = [int(c_hi[k:k + 2], 16) for k in (1, 3, 5)]
    return '#' + ''.join(f'{round(a[j] + (b[j] - a[j]) * t):02X}' for j in range(3))


_RAINBOW = [(0, '#2962FF'), (18, '#00B0FF'), (30, '#00E5FF'), (42, '#00E676'), (55, '#C6FF00'),
            (65, '#FFEA00'), (75, '#FF9100'), (87, '#FF3D00'), (100, '#FF1744')]


def rainbow_hex(value):
    """Same blue-to-red rainbow as f_rainbow in jojo.pine; returns '#RRGGBB' (None for na)."""
    if value is None or np.isnan(value):
        return None
    x = max(0.0, min(100.0, float(value)))
    for (lo, c_lo), (hi, c_hi) in zip(_RAINBOW[:-1], _RAINBOW[1:]):
        if x < hi or hi == 100:
            return _gradient(x, lo, hi, c_lo, c_hi)


# ---------------------------------------------------------------- the indicator
def components(df):
    """Compute the 14 components used by the current jojo.pine approximation."""
    o, h, l, c = (df[k].to_numpy(float) for k in ('open', 'high', 'low', 'close'))
    v = df['volume'].to_numpy(float) if 'volume' in df else np.full(len(c), np.nan)
    hlc3 = (h + l + c) / 3
    C = {}
    with np.errstate(divide='ignore', invalid='ignore'):
        # A. Components inferred from controlled probes
        C['roc'] = _clamp100(50.0 + 100.0 * (c - _shift(c, 10)) / _shift(c, 10))
        tp_chg = _change(hlc3)
        up_f = _sum(np.where(tp_chg > 0, v * hlc3, 0.0), 14)          # na change -> 0.0, as in Pine
        dn_f = _sum(np.where(tp_chg < 0, v * hlc3, 0.0), 14)
        tot = up_f + dn_f
        C['mfi'] = np.where(np.nan_to_num(tot, nan=-1.0) > 0, 100.0 * up_f / np.where(tot > 0, tot, 1.0), 50.0)
        mfv = np.where(h == l, 0.0, ((c - l) - (h - c)) / np.where(h == l, 1.0, h - l) * v)
        mfv_sum, vol_sum = _sum(mfv, 21), _sum(v, 21)
        cmf_ok = np.nan_to_num(vol_sum, nan=-1.0) > 0
        C['cmf'] = np.where(cmf_ok, _clamp100(50.0 + 150.0 * mfv_sum / np.where(cmf_ok, vol_sum, 1.0)), 50.0)
        # B. fitted price oscillators
        C['psy'] = _sum((_nz(_change(c)) > 0).astype(float), 14) / 14.0 * 100.0
        C['aroon_up'] = 100.0 * (_highestbars(h, 15) + 14) / 14.0
        C['aroon_dn'] = 100.0 - 100.0 * (_lowestbars(l, 15) + 14) / 14.0
        r7 = _rsi(c, 7)
        C['srsi'] = _stoch(r7, r7, r7, 14)
        C['bb'] = _clamp100(50.0 + 16.8176 * (hlc3 - _sma(hlc3, 22)) / _stdev(hlc3, 22))
        C['k17'] = _clamp100(50.0 + 0.9925 * (_stoch(c, h, l, 17) - 50.0))
        C['slope'] = _clamp100(50.0 + 82.02 * _slope_z(c, 12))
        C['tsi'] = _clamp100(50.0 + 73.96 * _tsi(c, 10, 9))
        C['fish'] = _clamp100(50.0 + 23.10 * _fisher(h, l, 14))
        C['rci'] = 50.0 + _rci(c, 9) / 2.0
        mom = c - _shift(c, 4)
        rmi_up = _rma(np.where(np.isnan(mom), np.nan, np.maximum(mom, 0)), 2)
        rmi_dn = _rma(np.where(np.isnan(mom), np.nan, np.maximum(-mom, 0)), 2)
        C['rmi'] = 100.0 * rmi_up / (rmi_up + rmi_dn)
    return C


def compute_jojo(df, with_components=False, **params):
    """Return a DataFrame with jojo, state code/text, bars in state, arrows and colour for every bar.

    df: columns open, high, low, close[, volume] (case-insensitive; 'Volume' from TradingView works).
    params: override any of DEFAULTS (ma_len, low_th, high_th, ob_level, ob_exit, os_level, os_exit).
    """
    p = {**DEFAULTS, **params}
    df = df.rename(columns={col: col.lower() for col in df.columns})
    C = components(df)
    total = W_ROC * C['roc'] + W_MFI * C['mfi'] + W_CMF * C['cmf'] + sum(w * C[k] for k, w in W_FIT.items())
    jojo = _clamp100(total)

    # state machine: +2 overbought, +1 recovered from overbought, -1 recovered from oversold, -2 oversold, 0 none yet
    n = len(jojo)
    state = np.zeros(n, dtype=int)
    bars_in_state = np.zeros(n, dtype=int)
    st, cnt = 0, 0
    for i in range(n):
        x = jojo[i]
        if not np.isnan(x):
            if x >= p['ob_level']:
                st = 2
            elif x <= p['os_level']:
                st = -2
            elif st == 2 and x < p['ob_exit']:
                st = 1
            elif st == -2 and x > p['os_exit']:
                st = -1
        cnt = 1 if (i == 0 or st != state[i - 1]) else cnt + 1
        state[i], bars_in_state[i] = st, cnt

    # arrows (ta.crossover / ta.crossunder of jojo vs its SMA)
    ma = _sma(jojo, p['ma_len'])
    j1, m1 = _shift(jojo), _shift(ma)
    buy = (jojo > ma) & (j1 <= m1) & (j1 < p['low_th'])
    sell = (jojo < ma) & (j1 >= m1) & (j1 > p['high_th'])

    out = pd.DataFrame({'jojo': jojo, 'state': state, 'state_text': [STATE_TEXT[s] for s in state],
                        'bars_in_state': bars_in_state, 'state_changed': np.r_[False, state[1:] != state[:-1]],
                        'buy': buy, 'sell': sell, 'color': [rainbow_hex(x) for x in jojo]}, index=df.index)
    if 'time' in df:
        out.insert(0, 'time', df['time'].to_numpy())
    if with_components:
        for k, x in C.items():
            out[f'c_{k}'] = x
    return out


# ---------------------------------------------------------------- command line
def _main(argv=None):
    ap = argparse.ArgumentParser(description='Compute jojo from an OHLCV CSV (e.g. TradingView "Export chart data").')
    ap.add_argument('csv', help='input CSV with time/open/high/low/close[/volume] columns')
    ap.add_argument('-o', '--out', help='write all bars to this CSV')
    ap.add_argument('--tail', type=int, default=10, help='number of last bars to print (default 10)')
    ap.add_argument('--components', action='store_true', help='also output the 14 components')
    ap.add_argument('--check', metavar='COLUMN', help='compare with a jojo column exported from TradingView')
    args = ap.parse_args(argv)

    raw = pd.read_csv(args.csv)
    lower = {c.lower(): c for c in raw.columns}
    missing = [k for k in ('open', 'high', 'low', 'close') if k not in lower]
    if missing:
        sys.exit(f'missing columns: {missing}')
    df = pd.DataFrame({k: raw[lower[k]] for k in ('time', 'open', 'high', 'low', 'close', 'volume') if k in lower})
    out = compute_jojo(df, with_components=args.components)
    if 'time' in out:
        out['date'] = pd.to_datetime(out['time'], unit='s', utc=True).dt.strftime('%Y-%m-%d')

    if args.check:
        ref = raw[args.check].to_numpy(float)
        m = np.isfinite(ref) & np.isfinite(out['jojo'].to_numpy())
        d = np.abs(out['jojo'].to_numpy()[m] - ref[m])
        print(f'check vs "{args.check}": {m.sum()} bars, max |diff| = {d.max():.3e}, last 100 bars max = {d[-100:].max():.3e}')

    cols = [c for c in ('date', 'jojo', 'state', 'state_text', 'bars_in_state', 'buy', 'sell') if c in out]
    with pd.option_context('display.width', 120, 'display.float_format', '{:.2f}'.format):
        print(out[cols].tail(args.tail).to_string(index=False))
    if args.out:
        out.to_csv(args.out, index=False)
        print(f'written: {args.out}')


if __name__ == '__main__':
    _main()
