"""Mean and std of test accuracy over seeds, per setup and question type.

Reads the slurm logs from Test.sh (avst-*.out) and Ablation.sh (ablate-*.out). The setup is
the --checkpoint name without _seedN, e.g. avst_dinov2_seed2 -> avst_dinov2, plus the
ablation if there is one, e.g. avst_dinov2+visual_zero. A log can hold several test runs.

    python AVST/tools/summarize_results.py avst-*.out
    python AVST/tools/summarize_results.py avst-*.out --latex
"""
import argparse
import re
import statistics
from collections import defaultdict

ACC = re.compile(r"^(\S.*?) Accuracy: ([\d.]+) %")
CKPT = re.compile(r"checkpoint='([^']+)'")
ABL = re.compile(r"ablate='([^']+)'")


def read_log(path):
    # the test runs: each starts at a Namespace(... mode='test' ...) line
    runs, setup, accs = [], None, {}
    for line in open(path, encoding='utf-8', errors='replace'):
        if line.startswith('Namespace(') and "mode='test'" in line:
            if setup and 'Overall' in accs:
                runs.append((setup, accs))
            setup, accs = CKPT.search(line).group(1), {}
            abl = ABL.search(line)
            if abl and abl.group(1) != 'none':
                setup += '+' + abl.group(1)
        elif setup:
            m = ACC.match(line.strip())
            if m:
                accs[m.group(1)] = float(m.group(2))
    if setup and 'Overall' in accs:
        runs.append((setup, accs))
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logs', nargs='+')
    parser.add_argument('--latex', action='store_true', help='print a LaTeX table body')
    args = parser.parse_args()

    runs = defaultdict(list)   # setup -> list of {type: acc}
    types = []
    for path in args.logs:
        res = read_log(path)
        if not res:
            print('skipped (no finished test run):', path)
        for ckpt, accs in res:
            runs[re.sub(r'_seed\d+', '', ckpt)].append(accs)
            print('%-40s %-44s overall %.2f' % (path, ckpt, accs['Overall']))
            types += [t for t in accs if t not in types]

    setups = sorted(runs)
    print('\nmean +- std over seeds, number of seeds:', {s: len(runs[s]) for s in setups}, '\n')

    def cell(s, t):
        vals = [r[t] for r in runs[s] if t in r]
        if not vals:
            return '-'
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return '%.2f +- %.2f' % (statistics.mean(vals), sd)

    if args.latex:
        end = ' ' + chr(92) * 2     # LaTeX row end
        print(' & '.join(['Question type'] + setups) + end)
        for t in types:
            print(' & '.join([t] + [cell(s, t).replace('+-', r'$\pm$') for s in setups]) + end)
    else:
        print('%-22s' % 'Question type' + ''.join('%-20s' % s for s in setups))
        for t in types:
            print('%-22s' % t + ''.join('%-20s' % cell(s, t) for s in setups))


if __name__ == '__main__':
    main()
