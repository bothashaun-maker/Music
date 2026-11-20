"""Integration check against a running local patched Studio; no model inference.

Uses a synthetic 75 MiB WAV, real native library/import APIs, and a deliberately
lost chunk acknowledgement. The optional Create-screen fixture only changes the
browser's readiness response; it never represents successful generation.
"""
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import urllib.error
import urllib.request
import wave

from playwright.sync_api import sync_playwright


def check(base='http://127.0.0.1:8791', key='local-test-key'):
    def call(path, *, method='GET', data=None, headers=None):
        payload = json.dumps(data).encode() if data is not None else None
        request = urllib.request.Request(base+path, method=method, data=payload,
                                         headers={'Content-Type': 'application/json', **(headers or {})})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    try:
        call('/health', headers={'X-Forwarded-For': '203.0.113.2'})
    except urllib.error.HTTPError as error:
        assert error.code in (401, 403)
    else:
        raise AssertionError('Forwarded unauthenticated request was accepted')
    assert call('/health', headers={'X-Forwarded-For': '203.0.113.2', 'Authorization': 'Bearer '+key})['status'] == 'ok'
    rpc = call('/mcp', method='POST', data={'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list', 'params': {}})
    tools = rpc['result']['tools']
    assert len(tools) > 50
    with tempfile.TemporaryDirectory(prefix='music-studio-browser-') as temporary:
        path = Path(temporary) / 'music-studio-upload-test.wav'
        period = array('h')
        for n in range(48000):
            sample = int(math.sin(2*math.pi*220*n/48000)*500)
            period.extend([sample, sample])
        if sys.byteorder != 'little':
            period.byteswap()
        with wave.open(str(path), 'wb') as output:
            output.setnchannels(2)
            output.setsampwidth(2)
            output.setframerate(48000)
            for _ in range(409):
                output.writeframesraw(period.tobytes())
            output.writeframesraw(period.tobytes()[:28800*4])
        original = path.read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        errors = []
        dropped = []
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=os.environ.get('CHROMIUM_PATH', '/usr/bin/chromium'), headless=True,
                args=['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
            page = browser.new_page(viewport={'width': 1440, 'height': 1000})
            page.on('pageerror', lambda error: errors.append(str(error)))

            def lose_ack(route):
                request = route.request
                if request.method == 'POST' and '?offset=4194304&' in request.url and not dropped:
                    response = route.fetch()
                    assert response.status == 200
                    dropped.append(True)
                    route.abort('failed')
                else:
                    route.continue_()

            page.route('**/kaggle/uploads/**', lose_ack)
            page.goto(base+'/kaggle-upload.html')
            page.locator('#file').set_input_files(path)
            page.locator('#send').click()
            page.wait_for_function("document.getElementById('status').textContent.includes('Ready in Library')", timeout=120000)
            assert dropped
            songs = call('/v1/library/songs')
            song = next(s for s in songs if s['metadata'].get('source_sha256') == digest)
            saved = Path(song['audio_path'])
            assert saved.read_bytes() == original
            count = len(songs)
            page.locator('#send').click()
            page.wait_for_function("document.getElementById('status').textContent.includes('Already in your Music Studio library')", timeout=30000)
            assert len(call('/v1/library/songs')) == count
            page.goto(base)
            page.get_by_role('button', name='Library', exact=True).click(timeout=30000)
            page.get_by_text('music-studio-upload-test', exact=True).first.wait_for(timeout=10000)
            page.get_by_role('button', name='Studio tools', exact=True).click()
            page.wait_for_timeout(500)
            if os.environ.get('MUSIC_STUDIO_SCREENSHOT'):
                page.screenshot(path=os.environ['MUSIC_STUDIO_SCREENSHOT'], full_page=True)
            # UI-only fixture lets us inspect controls on a machine without models.
            def readiness(route):
                response = route.fetch()
                data = response.json()
                data.update(ready=True, engine_ready=True, first_run=False)
                route.fulfill(response=response, json=data)
            page.route('**/setup/status', readiness)
            page.reload()
            page.get_by_role('button', name='Create', exact=True).click()
            page.locator('textarea').first.wait_for(timeout=15000)
            browser.close()
        assert not errors, errors
    return dict(native_library_import=True, source_bytes=len(original), source_sha256=digest,
                lost_ack_retry=True, repeat_import_idempotent=True, forwarded_auth_guard=True,
                mcp_tools=len(tools), javascript_errors=errors, create_screen='readiness fixture; no inference')


if __name__ == '__main__':
    print(json.dumps(check(os.environ.get('MUSIC_STUDIO_TEST_URL', 'http://127.0.0.1:8791'),
                           os.environ.get('MUSIC_STUDIO_TEST_KEY', 'local-test-key')), indent=2))
