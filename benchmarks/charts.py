"""Render saved benchmark measurements as Matplotlib PNGs."""
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'tiny-vllm-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COLORS = ['#475569', '#0d9488', '#2563eb']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'figure.facecolor': 'white',
                     'savefig.facecolor': 'white'})


def save(fig, path, note):
    path = Path(path).with_suffix('.png')
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.text(0.02, 0.015, note, fontsize=8, color='#475569')
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def paired(rows, labels, series, title, path, note, xlabel='Milliseconds per call (median)'):
    fig, ax = plt.subplots(figsize=(9.5, 5))
    y = np.arange(len(labels))
    height = 0.7 / len(series)
    maximum = max(float(values[i]) for _, values in series for i in range(len(labels)))
    for index, (label, values) in enumerate(series):
        pos = y - 0.35 + height * (index + 0.5)
        bars = ax.barh(pos, values, height=height * 0.88, label=label, color=COLORS[index])
        ax.bar_label(bars, labels=[f'{v:.3f}' for v in values], padding=4, fontsize=8)
    ax.set(yticks=y, yticklabels=labels, xlabel=xlabel,
           title=title, xlim=(0, maximum * 1.24))
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis='x', alpha=0.16)
    ax.legend(loc='lower right', frameon=False)
    save(fig, path, note)


def plot_kv(rows, path):
    prefixes = sorted({r['shared_prefix_tokens'] for r in rows})
    series = []
    for caching in (False, True):
        series.append(('Cache on' if caching else 'Cache off',
                       [next(r['median_ms'] for r in rows if r['caching'] == caching and r['shared_prefix_tokens'] == p)
                        for p in prefixes]))
    paired(rows, [f'{p} shared tokens' for p in prefixes], series, 'KV-cache bookkeeping · CPU', path,
           '1,000 requests/run · 1,024 tokens/request · 9 runs · metadata only, no model execution',
           xlabel='Milliseconds per 1,000-request run (median)')


def plot_scheduler(rows, path):
    seqs = sorted({r['max_num_seqs'] for r in rows})
    budgets = sorted({r['token_budget'] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, key, ylabel in [(axes[0], 'steps', 'Scheduling rounds'),
                            (axes[1], 'median_wall_ms', 'Simulation time (ms)')]:
        for i, seq in enumerate(seqs):
            values = [next(r[key] for r in rows if r['max_num_seqs'] == seq and r['token_budget'] == b)
                      for b in budgets]
            bars = ax.bar(np.arange(len(budgets)) + (i - 1) * 0.25, values,
                          width=0.23, color=COLORS[i], label=f'{seq} active seq')
            ax.bar_label(bars, labels=[f'{v:.0f}' if key == 'steps' else f'{v:.2f}' for v in values],
                         padding=3, fontsize=7)
        ax.set(xticks=np.arange(len(budgets)), xticklabels=budgets,
               xlabel='Token budget per round', ylabel=ylabel)
        ax.set_ylim(0, max(r[key] for r in rows) * 1.3)
        ax.grid(axis='y', alpha=0.16)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Scheduler · CPU', fontweight='bold')
    save(fig, path, '128 simulated requests · 32 synthetic outputs/request · 9 runs · no model execution')


def plot_attention(rows, device, path):
    selected = [r for r in rows if r['device'] == device]
    paired(selected, [r['workload'] for r in selected],
           [('Paged reference', [r['paged_median_ms'] for r in selected]),
            ('Dense SDPA control', [r['dense_median_ms'] for r in selected])],
           f'Reference attention · {device.upper()}', path,
           'Float32 · 8 query / 2 KV heads · 7 samples, 5 calls/sample · attention only')


def plot_model(rows, path):
    labels = list(dict.fromkeys(r['workload'] for r in rows))
    series = [(device.upper(), [next(r['median_ms'] for r in rows if r['device'] == device and r['workload'] == name)
                               for name in labels]) for device in dict.fromkeys(r['device'] for r in rows)]
    paired(rows, labels, series, 'Model forward pass · random weights', path,
           '3.54M parameters · float32 · 7 samples, 5 calls/sample · fixed-context replay, no text generation')


def plot_sampling(rows, path):
    labels = list(dict.fromkeys(r['workload'] for r in rows))
    series = [(device.upper(), [next(r['median_ms'] for r in rows if r['device'] == device and r['workload'] == name)
                               for name in labels]) for device in dict.fromkeys(r['device'] for r in rows)]
    paired(rows, labels, series, 'Token sampling · synthetic logits', path,
           '32,768-token vocabulary · float32 · 7 samples, 5 calls/sample · includes returned token IDs')


def plot_loading(rows, path):
    paired(rows, ['Local checkpoint'], [(r['device'].upper(), [r['median_ms']]) for r in rows],
           'Checkpoint loading · warm filesystem cache', path,
           'Local random-weight fixture · config, CPU model construction, weight copies, device transfer · 7 loads')


def main():
    output = Path('benchmarks/results')
    for name, render in [('kv_cache', plot_kv), ('scheduler', plot_scheduler), ('model', plot_model),
                         ('sampling', plot_sampling), ('loading', plot_loading)]:
        file = output / f'{name}.json'
        if file.exists():
            render(json.loads(file.read_text())['rows'], output / f'{name}.png')
    file = output / 'attention.json'
    if file.exists():
        data = json.loads(file.read_text())
        for device in data['devices']:
            plot_attention(data['rows'], device, output / f'attention_{device}.png')
    print('Matplotlib charts saved in benchmarks/results')


if __name__ == '__main__':
    main()
