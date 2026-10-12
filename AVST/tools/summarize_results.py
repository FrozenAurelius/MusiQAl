"""Summarise per-question prediction files (from main_avst.py --pred_out) into the
README results-table layout: accuracy per modality x question type, mean +- std over runs.

Usage:
    python AVST/tools/summarize_results.py results/resnet18_seed*.json
    python AVST/tools/summarize_results.py --label "ResNet-18" results/resnet18_seed*.json --csv out.csv
"""
import argparse
import csv
import json
import statistics

MODALITIES = ['Audio', 'Visual', 'Audio-Visual']
QTYPES = ['Existential', 'Location', 'Counting', 'Comparative', 'Temporal', 'Causal', 'Purpose']

# AVST row of the MusiQAl README results table (mean over runs).
PAPER_AVST = {
    'Audio': {'Existential': 76.11, 'Counting': 69.79, 'Comparative': 72.65, 'Temporal': 76.37,
              'Causal': 80.00, 'Avg': 72.32},
    'Visual': {'Existential': 94.55, 'Location': 69.08, 'Counting': 68.75, 'Temporal': 73.41,
               'Causal': 63.51, 'Avg': 69.48},
    'Audio-Visual': {'Existential': 61.19, 'Location': 66.10, 'Counting': 69.05, 'Comparative': 73.68,
                     'Temporal': 70.55, 'Causal': 64.40, 'Purpose': 65.97, 'Avg': 66.23},
    'AvgAll': 68.21,
}


def accuracies(records):
    """Accuracy (%) per (modality, qtype), per modality ('Avg') and overall, pooled over questions."""
    acc = {}
    for m in MODALITIES:
        rows = [r for r in records if r['modality'] == m]
        acc[m] = {}
        for q in QTYPES:
            sub = [r['correct'] for r in rows if r['qtype'] == q]
            if sub:
                acc[m][q] = 100.0 * sum(sub) / len(sub)
        if rows:
            acc[m]['Avg'] = 100.0 * sum(r['correct'] for r in rows) / len(rows)
    acc['AvgAll'] = 100.0 * sum(r['correct'] for r in records) / len(records)
    return acc


def mean_std(values):
    if len(values) == 1:
        return values[0], float('nan')
    return statistics.mean(values), statistics.stdev(values)


def fmt(m, s):
    return '%6.2f' % m if s != s else '%6.2f ±%4.2f' % (m, s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', help='prediction json files, one per run/seed')
    parser.add_argument('--label', default='this run', help='name for the model row')
    parser.add_argument('--csv', default=None, help='optionally write the summary to a csv file')
    args = parser.parse_args()

    runs = [accuracies(json.load(open(f))) for f in args.files]
    n_q = len(json.load(open(args.files[0])))
    print('%s: %d run(s), %d test questions\n' % (args.label, len(runs), n_q))

    out_rows = []
    for m in MODALITIES:
        cols = [q for q in QTYPES + ['Avg'] if q in runs[0][m]]
        print('== %s ==' % m)
        print('%-14s' % 'question type' + ''.join('%15s' % c for c in ['ours', 'paper AVST', 'diff']))
        for q in cols:
            mu, sd = mean_std([r[m][q] for r in runs])
            ref = PAPER_AVST[m].get(q)
            diff = '%+6.2f' % (mu - ref) if ref is not None else ''
            print('%-14s%15s%15s%15s' % (q, fmt(mu, sd), '%.2f' % ref if ref is not None else '-', diff))
            out_rows.append([args.label, m, q, '%.2f' % mu, '' if sd != sd else '%.2f' % sd, ref])
        print()
    mu, sd = mean_std([r['AvgAll'] for r in runs])
    print('AvgAll: %s   (paper AVST %.2f, diff %+.2f)' % (fmt(mu, sd), PAPER_AVST['AvgAll'], mu - PAPER_AVST['AvgAll']))
    out_rows.append([args.label, 'All', 'AvgAll', '%.2f' % mu, '' if sd != sd else '%.2f' % sd, PAPER_AVST['AvgAll']])

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['model', 'modality', 'qtype', 'mean', 'std', 'paper_avst'])
            w.writerows(out_rows)
        print('wrote', args.csv)


if __name__ == '__main__':
    main()
