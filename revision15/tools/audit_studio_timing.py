"""Measure explicitly confirmed lyric-marker references; never infer ground truth."""
import argparse
import json
from pathlib import Path
import shlex
import statistics
import zipfile


def read_result(path):
    path = Path(path)
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as archive:
            matches = [n for n in archive.namelist()
                       if n.endswith('_timestamps.json') or n == 'timestamps.json']
            if len(matches) != 1:
                raise ValueError('Expected exactly one timestamp JSON in the ZIP')
            return json.loads(archive.read(matches[0]))
    return json.loads(path.read_text())


def project_markers(path):
    markers, depth = {}, 0
    for line in Path(path).read_text(errors='strict').splitlines():
        text = line.strip()
        if text.startswith('<'):
            depth += 1
        elif text == '>':
            depth -= 1
        elif depth == 1 and text.startswith('MARKER '):
            fields = shlex.split(text)
            if int(fields[4]) & 1:
                continue  # Region starts and ends share IDs with lyric markers.
            ident = int(fields[1])
            if ident in markers:
                raise ValueError('Duplicate project lyric marker ID: ' + str(ident))
            markers[ident] = {'text':fields[3], 'start':float(fields[2])}
    return markers


def measure(result, reference, ids):
    lines = [e for e in result['entries'] if e['kind'] == 'sing']
    rows = []
    for ident in ids:
        if ident not in reference or not 1 <= ident <= len(lines):
            raise ValueError('Missing reference or candidate lyric marker: ' + str(ident))
        line, truth = lines[ident-1], reference[ident]
        if line['text'] != truth['text']:
            raise ValueError('Lyric label/order differs at marker ' + str(ident))
        error = float(line['start'])-truth['start']
        rows.append({'marker':ident,'reference':truth['start'],'candidate':line['start'],
                     'error_seconds':error})
    if not rows:
        raise ValueError('Select explicitly confirmed reference marker IDs')
    errors = [abs(r['error_seconds']) for r in rows]
    return {'metric':'lyric marker start error in seconds', 'references':len(rows),
            'mean_absolute_error':statistics.mean(errors),
            'median_absolute_error':statistics.median(errors),'max_absolute_error':max(errors),
            'within_250ms':sum(e <= .25 for e in errors),'within_500ms':sum(e <= .5 for e in errors),
            'rows':rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--reference-markers', required=True, help='Confirmed IDs/ranges, e.g. 1:15 or 1:68,70:87')
    args = parser.parse_args()
    ids = set()
    for part in args.reference_markers.split(','):
        bounds = list(map(int,part.split(':')))
        if len(bounds) not in (1,2) or bounds[0] < 1 or bounds[-1] < bounds[0]:
            parser.error('Invalid confirmed marker ID/range: ' + part)
        ids.update(range(bounds[0],bounds[-1]+1))
    report = measure(read_result(args.candidate), project_markers(args.project), sorted(ids))
    print(json.dumps(report,indent=2))
