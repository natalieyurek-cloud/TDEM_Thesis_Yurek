"""
TEM Groundwater Detection - Pulse Processing Pipeline
======================================================
Steps:
  1. Chop WAV into per-pulse segments — detects flyback via negative peak finding,
     with autocorrelation-based period estimation for robust thresholding.
     Works with int16/int32, mono/stereo, any sample rate.
  2. Average all segments together with cross-correlation time alignment.
  3. Compute logarithmically-spaced gates over the averaged decay.

Usage:
  python tem_processor.py                            # uses defaults
  python tem_processor.py --input HoverTEM.wav       # custom input file
  python tem_processor.py --gates 20                 # custom gate count
  python tem_processor.py --channel 1                # use right stereo channel
  python tem_processor.py --decay-end 90             # trim gate window (samples)
  python tem_processor.py --no-plot                  # skip diagnostic plots
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate, find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import json
import os

# ─────────────────────────────────────────────
# CONFIG / DEFAULTS
# ─────────────────────────────────────────────
# DEFAULT_INPUT and OUTPUT_DIR should be the only things that need to be changed
DEFAULT_INPUT     = "C:/Users/yetfl/SpaceGrant/HoverTEM/Audio/WAV_Processing/Raw/Field_3-6/correctlydamped.wav"
DEFAULT_CHANNEL   = 1       # stereo channel to use (ignored for mono)
DEFAULT_GATES     = 20      # number of log-spaced gates
DEFAULT_DECAY_END = None    # if None, auto (full period minus a small guard)
DEFAULT_PLOT      = True
OUTPUT_DIR        = "C:/Users/yetfl/SpaceGrant/HoverTEM/Audio/WAV_Processing/Results/Field_3-6/correctlydamped"


# ─────────────────────────────────────────────
# LOAD & NORMALISE
# ─────────────────────────────────────────────
def load_signal(path, channel=0):
    sr, data = wavfile.read(path)

    if data.ndim == 2:
        sig_raw = data[:, channel].astype(float)
        n_ch = data.shape[1]
    else:
        sig_raw = data.astype(float)
        n_ch = 1

    dtype_max = np.iinfo(data.dtype).max
    norm = sig_raw / dtype_max

    print(f"[Load] {os.path.basename(path)}")
    print(f"       {sr} Hz | {data.dtype} | "
          f"{'stereo' if n_ch > 1 else 'mono'} | "
          f"{len(norm)/sr:.2f}s | channel {channel if n_ch > 1 else 'N/A (mono)'}")

    return sr, norm, sig_raw


# ─────────────────────────────────────────────
# STEP 1A — ESTIMATE PERIOD VIA AUTOCORRELATION
# ─────────────────────────────────────────────
def estimate_period(norm, sr, ac_window=4096, search_min=20, search_max=500):
    chunk = norm[:int(sr * 0.5)]
    chunk_z = chunk - chunk.mean()
    ac = np.correlate(chunk_z[:ac_window], chunk_z[:ac_window], mode='full')
    ac = ac[len(ac) // 2:]
    ac /= ac[0]

    peaks, _ = find_peaks(ac[search_min:search_max], height=0.3)
    if len(peaks) == 0:
        raise RuntimeError(
            "Could not detect a dominant pulse period via autocorrelation. "
            "Check that the signal contains regular TEM pulses."
        )

    period = int(peaks[0]) + search_min
    print(f"\n[Step 1] Autocorr period estimate: {period} samples "
          f"({period / sr * 1000:.3f} ms  ->  {1 / (period / sr):.1f} Hz)")
    return period


# ─────────────────────────────────────────────
# STEP 1B — DETECT FLYBACK PEAKS
# ─────────────────────────────────────────────
def find_flyback_peaks(norm, sr, period_samples,
                       peak_height_fraction=0.2,
                       min_dist_fraction=0.7):
    min_dist = int(period_samples * min_dist_fraction)

    flybacks, _ = find_peaks(-norm, distance=min_dist,
                              height=peak_height_fraction)

    if len(flybacks) < 2:
        raise RuntimeError(
            f"Only {len(flybacks)} flyback peaks detected. "
            "Try lowering --peak-height or check the signal."
        )

    diffs = np.diff(flybacks)
    print(f"       Flyback peaks found: {len(flybacks)}")
    print(f"       Period -- mean: {diffs.mean() / sr * 1000:.4f} ms, "
          f"std: {diffs.std() / sr * 1000:.4f} ms")
    return flybacks


# ─────────────────────────────────────────────
# STEP 1C — CHOP INTO SEGMENTS
# ─────────────────────────────────────────────
def chop_segments(sig_raw, flybacks, tolerance=2):
    lengths = np.diff(flybacks)
    median_len = int(np.median(lengths))

    segments = []
    discarded = 0
    for i in range(len(flybacks) - 1):
        seg = sig_raw[flybacks[i]: flybacks[i + 1]]
        if abs(len(seg) - median_len) <= tolerance:
            segments.append(seg.astype(float))
        else:
            discarded += 1

    print(f"       Median segment length: {median_len} samples "
          f"({median_len / 44100 * 1000:.3f} ms approx)")
    print(f"       Kept {len(segments)} segments, discarded {discarded} outliers")
    return segments, median_len


# ─────────────────────────────────────────────
# STEP 2 — ALIGN & AVERAGE SEGMENTS
# ─────────────────────────────────────────────
def align_and_average(segments, median_len, max_shift=5):
    """
    Cross-correlate each segment against the first (reference) to find
    any sub-period timing jitter, shift to align, then stack and average.
    Also accumulates a sum-of-squares array so per-sample std can be
    computed and passed downstream for gate-level error bars.
    """
    reference = np.array(segments[0])
    stack    = np.zeros(median_len, dtype=float)
    stack_sq = np.zeros(median_len, dtype=float)   # sum of squares for std
    counts   = np.zeros(median_len, dtype=int)      # how many segments contributed
    shifts   = []

    for seg in segments:
        s = np.array(seg, dtype=float)
        corr = correlate(s, reference, mode='full')
        lag = int(corr.argmax()) - (len(reference) - 1)
        lag = int(np.clip(lag, -max_shift, max_shift))
        shifts.append(lag)

        if lag > 0:
            aligned = np.concatenate([np.zeros(lag), s[:-lag] if lag < len(s) else []])
        elif lag < 0:
            aligned = np.concatenate([s[-lag:], np.zeros(-lag)])
        else:
            aligned = s

        n = min(len(aligned), median_len)
        stack[:n]    += aligned[:n]
        stack_sq[:n] += aligned[:n] ** 2
        counts[:n]   += 1

    # Avoid divide-by-zero for any unfilled samples
    counts = np.maximum(counts, 1)
    averaged = stack / counts

    # Per-sample standard deviation: std = sqrt(E[x²] - E[x]²)
    sample_std = np.sqrt(np.maximum(stack_sq / counts - averaged ** 2, 0))

    shifts = np.array(shifts)
    print(f"\n[Step 2] Averaged {len(segments)} segments")
    print(f"         Shift stats -- mean: {shifts.mean():.3f} samples, "
          f"std: {shifts.std():.3f}, max abs: {np.abs(shifts).max()} samples")
    return averaged, sample_std, shifts


# ─────────────────────────────────────────────
# STEP 3 — LOGARITHMIC GATES
# ─────────────────────────────────────────────
def compute_gates(averaged, sample_std, sample_rate, n_gates,
                  flyback_sample=0, decay_end=None):
    """
    Compute log-spaced time gates over the decay portion of the averaged pulse.
    Also computes per-gate std as the mean of per-sample stds within the gate
    (conservative — treats each sample's std independently).
    """
    if decay_end is None:
        decay_end = len(averaged) - 3
    decay_end = min(decay_end, len(averaged) - 1)

    n_decay = decay_end - flyback_sample
    if n_decay < n_gates:
        raise ValueError(
            f"Only {n_decay} decay samples available for {n_gates} gates. "
            "Reduce --gates or increase --decay-end."
        )

    log_edges = np.unique(
        np.round(
            np.logspace(
                np.log10(flyback_sample + 1),
                np.log10(decay_end),
                n_gates + 1
            )
        ).astype(int)
    )
    log_edges = np.clip(log_edges, flyback_sample + 1, decay_end)

    while len(log_edges) < n_gates + 1:
        log_edges = np.unique(np.append(log_edges, log_edges[-1] + 1))
    log_edges = log_edges[:n_gates + 1]

    gates = []
    for i in range(n_gates):
        i0 = int(log_edges[i])
        i1 = int(log_edges[i + 1]) - 1
        i1 = min(i1, decay_end)

        t0_ms = (i0 - flyback_sample) / sample_rate * 1000
        t1_ms = (i1 - flyback_sample) / sample_rate * 1000
        tc_ms = (np.sqrt(t0_ms * t1_ms)
                 if (t0_ms > 0 and t1_ms > 0)
                 else (t0_ms + t1_ms) / 2)

        window     = averaged[i0: i1 + 1]
        window_std = sample_std[i0: i1 + 1]
        if len(window) == 0:
            continue

        gates.append({
            "gate_number":    i + 1,
            "start_sample":   i0,
            "end_sample":     i1,
            "start_time_ms":  round(t0_ms, 6),
            "end_time_ms":    round(t1_ms, 6),
            "center_time_ms": round(tc_ms, 6),
            "n_points":       len(window),
            "mean_value":     float(np.mean(window)),
            "std_value":      float(np.mean(window_std)),   # mean of per-sample stds
            "rms_value":      float(np.sqrt(np.mean(window ** 2))),
        })

    print(f"\n[Step 3] Computed {len(gates)} log-spaced gates")
    print(f"         Time range: {gates[0]['start_time_ms']:.4f} ms "
          f"-> {gates[-1]['end_time_ms']:.4f} ms")
    print(f"         Points per gate: "
          f"min={min(g['n_points'] for g in gates)}, "
          f"max={max(g['n_points'] for g in gates)}")
    return gates


# ─────────────────────────────────────────────
# DIAGNOSTICS / PLOTS
# ─────────────────────────────────────────────
def plot_diagnostics(norm, sig_raw, flybacks, segments,
                     averaged, sample_std, gates, sample_rate, input_path, output_dir):
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    basename = os.path.splitext(os.path.basename(input_path))[0]

    # 1. Raw signal first 50 ms with flyback markers
    n50 = int(sample_rate * 0.05)
    t50 = np.arange(n50) / sample_rate * 1000
    axes[0].plot(t50, norm[:n50], lw=0.7, color='steelblue')
    pks_win = flybacks[flybacks < n50]
    axes[0].plot(pks_win / sample_rate * 1000, norm[pks_win],
                 'rv', ms=6, label='Detected flyback', zorder=5)
    axes[0].axhline(0, color='gray', lw=0.5, ls='--')
    axes[0].set_title(f'[{basename}] Raw Signal — First 50 ms (normalised)')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude (normalised)')
    axes[0].legend()

    # 2. First 10 segments overlaid
    min_len = min(len(s) for s in segments[:10])
    t_seg = np.arange(min_len) / sample_rate * 1000
    for s in segments[:10]:
        axes[1].plot(t_seg, s[:min_len], lw=0.5, alpha=0.6)
    axes[1].axhline(0, color='red', lw=0.5, ls='--')
    axes[1].set_title('First 10 Segments Overlaid (raw amplitude)')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Amplitude')

    # 3. Averaged pulse with std shading + gate std error bars
    t_avg = np.arange(len(averaged)) / sample_rate * 1000
    axes[2].plot(t_avg, averaged, lw=1.2, color='darkgreen', label='Averaged pulse')
    axes[2].fill_between(t_avg,
                         averaged - sample_std,
                         averaged + sample_std,
                         alpha=0.25, color='darkgreen', label='±1 std (per sample)')

    # Gate boundaries + std error bars at gate centre
    for g in gates:
        axes[2].axvspan(g['start_time_ms'], g['end_time_ms'],
                        alpha=0.10, color='orange')
        axes[2].axvline(g['start_time_ms'], color='orange', lw=0.4, alpha=0.7)
        # Error bar: vertical line at gate centre ± gate std
        axes[2].errorbar(g['center_time_ms'], g['mean_value'],
                         yerr=g['std_value'],
                         fmt='none', color='crimson', capsize=3,
                         lw=1.2, alpha=0.8)

    axes[2].axhline(0, color='red', lw=0.5, ls='--')
    axes[2].set_title(f'Averaged Pulse — ±1 std shading & gate error bars')
    axes[2].set_xlabel('Time (ms)')
    axes[2].set_ylabel('Amplitude')
    axes[2].legend(loc='upper right')

    # 4. TEM decay curve with error bars
    valid_gates = [g for g in gates if g['mean_value'] != 0 and g['n_points'] > 0]
    if valid_gates:
        tc  = np.array([g['center_time_ms'] for g in valid_gates])
        mv  = np.array([abs(g['mean_value'])  for g in valid_gates])
        sv  = np.array([g['std_value']         for g in valid_gates])
        axes[3].loglog(tc, mv, 'o-', color='firebrick', lw=1.5, ms=5, zorder=3)
        # Asymmetric error bars clipped so lower bound never goes <= 0 on log axis
        lo = np.minimum(sv, mv * 0.99)
        axes[3].errorbar(tc, mv, yerr=[lo, sv],
                         fmt='none', color='firebrick',
                         capsize=4, lw=1.0, alpha=0.6)
    axes[3].set_title('TEM Decay Curve — Gate Mean Values ± std')
    axes[3].set_xlabel('Time (ms, log scale)')
    axes[3].set_ylabel('|Mean Amplitude| (log scale)')
    axes[3].grid(True, which='both', alpha=0.3)
    axes[3].set_ylim(10000000,10000000000)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{basename}_diagnostics.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n[Plot] Saved diagnostics -> {out_path}")
    return out_path


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="TEM Pulse Processor")
    parser.add_argument('--input',       default=DEFAULT_INPUT,
                        help='Input WAV file')
    parser.add_argument('--channel',     default=DEFAULT_CHANNEL, type=int,
                        help='Stereo channel (0=left, 1=right). Ignored for mono.')
    parser.add_argument('--gates',       default=DEFAULT_GATES, type=int,
                        help='Number of log-spaced gates')
    parser.add_argument('--decay-end',   default=DEFAULT_DECAY_END, type=int,
                        help='Last sample index for gate window (trims next-pulse bleed). '
                             'Default: auto (segment length - 3).')
    parser.add_argument('--peak-height', default=0.2, type=float,
                        help='Min normalised amplitude for flyback detection (default 0.2)')
    parser.add_argument('--no-plot',     action='store_true',
                        help='Skip diagnostic plots')
    args = parser.parse_args()

    print("=" * 55)
    print("  TEM Pulse Processor")
    print("=" * 55)

    # Load
    sr, norm, sig_raw = load_signal(args.input, args.channel)

    # Step 1 — detect flybacks & chop
    period_samples       = estimate_period(norm, sr)
    flybacks             = find_flyback_peaks(norm, sr, period_samples,
                                              peak_height_fraction=args.peak_height)
    segments, median_len = chop_segments(sig_raw, flybacks)

    # Step 2 — align & average (now also returns per-sample std)
    averaged, sample_std, shifts = align_and_average(segments, median_len)

    # Step 3 — gates (now also computes per-gate std)
    gates = compute_gates(averaged, sample_std, sr, args.gates,
                          decay_end=args.decay_end)

    # Print gate table
    print("\n Gate # |  Start (ms) |   End (ms)  | Center (ms) | N pts | Mean value |   Std")
    print(" -------+-------------+-------------+-------------+-------+------------+--------")
    for g in gates:
        print(f"  {g['gate_number']:4d}  | {g['start_time_ms']:11.4f} | "
              f"{g['end_time_ms']:11.4f} | {g['center_time_ms']:11.4f} | "
              f"{g['n_points']:5d} | {g['mean_value']:10.2f} | {g['std_value']:8.2f}")

    # Save results
    basename = os.path.splitext(os.path.basename(args.input))[0]
    results = {
        "input_file":          args.input,
        "sample_rate_hz":      sr,
        "n_segments":          len(segments),
        "segment_len_samples": median_len,
        "segment_len_ms":      round(median_len / sr * 1000, 6),
        "averaged_pulse":      averaged.tolist(),
        "sample_std":          sample_std.tolist(),
        "gates":               gates,
    }
    json_path = os.path.join(OUTPUT_DIR, f'{basename}_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Save] Results -> {json_path}")

    csv_path = os.path.join(OUTPUT_DIR, f'{basename}_averaged_pulse.csv')
    t_ms = np.arange(len(averaged)) / sr * 1000
    np.savetxt(csv_path, np.column_stack([t_ms, averaged]),
            delimiter=',', header='time_ms,amplitude', comments='')
    print(f"[Save] Averaged pulse -> {csv_path}")

    decay_path = os.path.join(OUTPUT_DIR, f'{basename}_decay_curve.csv')
    valid_gates = [g for g in gates if g['mean_value'] != 0 and g['n_points'] > 0]
    decay_rows = np.array([[g['center_time_ms'], abs(g['mean_value']), g['std_value']] 
                        for g in valid_gates])
    np.savetxt(decay_path, decay_rows,
            delimiter=',', header='center_time_ms,abs_mean_amplitude,std',
            comments='')
    print(f"[Save] Decay curve   -> {decay_path}")

    if not args.no_plot:
        plot_diagnostics(norm, sig_raw, flybacks, segments,
                         averaged, sample_std, gates, sr, args.input, OUTPUT_DIR)

    print("\nDone.")
    return averaged, gates


if __name__ == "__main__":
    main()