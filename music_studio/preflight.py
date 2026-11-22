"""Verify every pinned remote before the long build or a multi-gigabyte download.

The other automatic Dataset builders fail in seconds when a pin is wrong. This
one used to discover a dead URL only after compiling cargo, npm and CUDA for the
best part of an hour, so every remote is probed first and the run stops with a
readable list instead of a traceback from deep inside the build.
"""
from __future__ import annotations

import urllib.error
import urllib.request

UA = 'Music-Studio-Builder/1'
MARK = 'bytes 0-0/'


def probe(url, timeout=30):
    """Return (state, size) where state is 'ok', 'unreachable' or 'unknown'."""
    request = urllib.request.Request(url, headers={'User-Agent': UA}, method='HEAD')
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            length = response.headers.get('Content-Length')
            return ('ok', int(length)) if length and length.isdigit() else ('unknown', None)
    except urllib.error.HTTPError as error:
        if error.code not in (403, 405, 501):
            return 'unreachable', error.code
    except Exception as error:  # noqa: BLE001 - reported, never swallowed silently
        return 'unreachable', type(error).__name__
    # Some hosts refuse HEAD; ask for a single byte and read the total from the range.
    request = urllib.request.Request(url, headers={'User-Agent': UA, 'Range': 'bytes=0-0'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_range = response.headers.get('Content-Range', '')
            if '/' in content_range:
                total = content_range.rpartition('/')[2].strip()
                if total.isdigit():
                    return 'ok', int(total)
            return 'unknown', None
    except Exception as error:  # noqa: BLE001
        return 'unreachable', type(error).__name__


def remote_records(catalog_data, sources_data):
    """Every pinned remote this builder consumes, models first."""
    records = [dict(name=a['name'], source=a['source'], bytes=a['bytes'], kind='model')
               for a in catalog_data['assets']]
    tools = list(sources_data['runtime_assets']) + [sources_data['node'], sources_data['rustup'],
                                                    sources_data['cloudflared']]
    records += [dict(name=r['name'], source=r['source'], bytes=r['bytes'], kind='toolchain') for r in tools]
    return records


def preflight(records, timeout=30):
    """Probe each pinned remote and raise one readable error listing the failures."""
    failures, unknown = [], []
    print(f'Preflight: checking {len(records)} pinned downloads before any build step.', flush=True)
    for record in records:
        state, size = probe(record['source'], timeout=timeout)
        if state == 'ok' and size == record['bytes']:
            continue
        if state == 'unknown':
            unknown.append(record['name'])
            print(f'  ? {record["name"]}: server reported no length; size is verified on download.', flush=True)
        elif state == 'ok':
            failures.append(f'{record["name"]}: server has {size} bytes, pinned {record["bytes"]}'
                            f' ({record["source"]})')
        else:
            failures.append(f'{record["name"]}: unreachable ({size}) ({record["source"]})')
    if failures:
        raise RuntimeError('Preflight failed; nothing was compiled or downloaded:\n  ' + '\n  '.join(failures))
    print(f'Preflight: {len(records)} remotes reachable with pinned sizes'
          + (f'; {len(unknown)} verified during download.' if unknown else '.'), flush=True)
    return True
