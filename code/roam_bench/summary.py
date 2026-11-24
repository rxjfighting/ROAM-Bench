"""Summarize correctness, evidence checks, and strict task success over completed episodes."""
import argparse
import json
from pathlib import Path

def summarize(rows):
    count = len(rows)
    if count == 0:
        raise ValueError('No completed episodes.')
    counts = {metric: sum((row[metric] is True for row in rows)) for metric in ('A', 'N', 'S', 'T')}
    error_s = sum((row['S'] is True and row['A'] is False for row in rows))
    gap = sum(row['A'] is True and row['N'] is False for row in rows)
    actions = [row.get('score', {}).get('steps') for row in rows]
    known_actions = [value for value in actions if isinstance(value, int) and not isinstance(value, bool) and value >= 0]
    return {'episodes': count, 'counts': counts, 'rates': {key: value / count for key, value in counts.items()},
            'correctness_task_completion_gap': {'count': gap, 'rate': gap / count},
            'error_after_S': {'numerator': error_s, 'denominator': counts['S'], 'rate': error_s / counts['S'] if counts['S'] else None},
            'average_actions': sum(known_actions) / count if len(known_actions) == count else None,
            'episodes_with_action_counts': len(known_actions)}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args(argv)
    document = json.loads(args.results.read_text())
    print(json.dumps(summarize(document['results']), indent=2))
if __name__ == '__main__':
    main()
