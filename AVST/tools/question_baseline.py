# Question-only baselines for MusiQAl: answers without looking at audio or video.
#
#   python AVST/tools/question_baseline.py
#
# Q (Table II): for each test question, the most common training answer to the same question
#   (same template and same filled-in values). For test questions that never occur in the training
#   set, the most common training answer for that question type is used instead.
# Template: the most common training answer per question template only (values ignored), with the
#   most common answer overall as fallback. This is the kind of baseline Jensen et al. (2026) use.
# Ties go to the answer seen first in the training file.
import ast
import json
import argparse
from collections import Counter, defaultdict

# question types in the order the AVST test prints them
TYPES = [('Audio', 'Existential'), ('Audio', 'Counting'), ('Audio', 'Comparative'), ('Audio', 'Temporal'), ('Audio', 'Causal'),
         ('Visual', 'Existential'), ('Visual', 'Location'), ('Visual', 'Counting'), ('Visual', 'Temporal'), ('Visual', 'Causal'),
         ('Audio-Visual', 'Existential'), ('Audio-Visual', 'Counting'), ('Audio-Visual', 'Location'), ('Audio-Visual', 'Comparative'),
         ('Audio-Visual', 'Temporal'), ('Audio-Visual', 'Causal'), ('Audio-Visual', 'Purpose')]


def question_key(q):
    return q['question_content'].strip() + '|' + q['templ_values']


def template_key(q):
    return q['question_content'].strip()


def most_common(counter):
    return counter.most_common(1)[0][0]   # stable: ties keep the order answers were first seen


def evaluate(train, test, key, fallback):
    by_key = defaultdict(Counter)
    for q in train:
        by_key[key(q)][q['answer']] += 1
    correct = defaultdict(lambda: [0, 0])
    unseen = 0
    for q in test:
        if key(q) in by_key:
            pred = most_common(by_key[key(q)])
        else:
            unseen += 1
            pred = fallback(q)
        t = tuple(ast.literal_eval(q['type']))
        for group in (t, (t[0],), ('Overall',)):
            correct[group][0] += pred == q['answer']
            correct[group][1] += 1
    return correct, unseen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_train', default='AVST/data/json/avqa-train.json')
    parser.add_argument('--label_test', default='AVST/data/json/avqa-test.json')
    args = parser.parse_args()
    train = json.load(open(args.label_train))
    test = json.load(open(args.label_test))

    by_type = defaultdict(Counter)
    for q in train:
        by_type[q['type']][q['answer']] += 1
    overall = Counter(q['answer'] for q in train)

    q_res, q_unseen = evaluate(train, test, question_key, lambda q: most_common(by_type[q['type']]))
    t_res, t_unseen = evaluate(train, test, template_key, lambda q: most_common(overall))

    print('test questions: %d | not in the training set: %d (same question), %d (same template)\n'
          % (len(test), q_unseen, t_unseen))
    print('%-28s %5s %8s %10s' % ('Question type', 'n', 'Q', 'Template'))
    rows = TYPES[:5] + [('Audio',)] + TYPES[5:10] + [('Visual',)] + TYPES[10:] + [('Audio-Visual',), ('Overall',)]
    for g in rows:
        n = q_res[g][1]
        if n == 0:
            continue
        print('%-28s %5d %8.1f %10.1f' % (' '.join(g), n, 100 * q_res[g][0] / n, 100 * t_res[g][0] / n))


if __name__ == '__main__':
    main()
