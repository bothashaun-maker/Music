"""Build Music Studio's PDF in the existing Music Production System visual format."""
from pathlib import Path
import importlib.util
import json

from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('music_guide_style', ROOT / 'tools/build_music_production_system.py')
style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)


def base(c, page, total):
    c.setFillColor(style.BG)
    c.rect(0, 0, style.WIDTH, style.HEIGHT, fill=1, stroke=0)
    c.setStrokeColor(style.CORAL)
    c.setLineWidth(.9)
    for x, y, sx, sy in [(18, 18, 1, 1), (style.WIDTH-18, 18, -1, 1),
                          (18, style.HEIGHT-18, 1, -1), (style.WIDTH-18, style.HEIGHT-18, -1, -1)]:
        c.line(x, y, x+13*sx, y)
        c.line(x, y, x, y+13*sy)
    if page > 1:
        style.tracking(c, 'JESUSJUICEPRODUCTIONS', style.MARGIN, style.HEIGHT-39, 6.4, 1.15, style.CYAN)
        c.setFont('JJP', 6.8)
        c.setFillColor(style.MUTED)
        c.drawRightString(style.WIDTH-style.MARGIN, style.HEIGHT-39, 'MUSIC STUDIO  /  FIELD GUIDE')
        c.setStrokeColor(style.LINE)
        c.line(style.MARGIN, 48, style.WIDTH-style.MARGIN, 48)
        c.setFont('JJP', 7)
        c.drawString(style.MARGIN, 33, 'MUSIC STUDIO v1  ·  11 NOV 2026')
        c.drawRightString(style.WIDTH-style.MARGIN, 33, f'{page:02} / {total:02}')
        c.setFillColor(style.CYAN)
        c.drawRightString(style.WIDTH-style.MARGIN, style.HEIGHT-56, 'BACK TO STUDIO DESK')
        c.linkRect('', 'desk', (style.WIDTH-style.MARGIN-115, style.HEIGHT-60, style.WIDTH-style.MARGIN, style.HEIGHT-47), thickness=0)


def build():
    data = json.loads((ROOT / 'docs/music-studio/guide.json').read_text())
    style.fonts()
    target = ROOT / 'Music-Studio-Guide.pdf'
    c = canvas.Canvas(str(target), pagesize=(style.WIDTH, style.HEIGHT), invariant=1, pageCompression=1)
    c.setTitle('JesusJuiceProductions — Music Studio Field Guide · Edition I')
    c.setAuthor('JesusJuiceProductions')
    c.setSubject('Kaggle Music Studio: setup, songs, scores, MIDI, covers, stems, karaoke, processing, video, LoRA and MCP')
    total = len(data['pages']) + 1
    base(c, 1, total)
    c.bookmarkPage('cover')
    c.addOutlineEntry('Music Studio Field Guide', 'cover', 0)
    c.setStrokeColor(style.CYAN)
    c.roundRect(style.MARGIN, 584, 296, 24, 2, stroke=1, fill=0)
    style.tracking(c, data['brand'], style.MARGIN+12, 592, 7.1, 2, style.CYAN)
    for text, y, color in [('THE MUSIC', 523, style.WHITE), ('STUDIO', 471, style.WHITE), ('FIELD GUIDE', 419, style.CORAL)]:
        c.setFillColor(color)
        c.setFont('JJP-Bold', 43)
        c.drawString(style.MARGIN, y, text)
    paragraph = Paragraph(style.safe(data['subtitle']), ParagraphStyle('sub', fontName='JJP', fontSize=11.1, leading=18.5, textColor=style.MUTED))
    _, height = paragraph.wrap(style.CONTENT_WIDTH, 100)
    paragraph.drawOn(c, style.MARGIN, 370-height)
    c.setFillColor(style.PANEL)
    c.roundRect(style.MARGIN, 205, style.CONTENT_WIDTH, 72, 3, fill=1, stroke=0)
    quote = Paragraph('“Find the song. Shape the performance.<br/>Keep the work you want to return to.”', ParagraphStyle('quote', fontName='JJP-Italic', fontSize=12.4, leading=19, textColor=style.WHITE))
    quote.wrap(style.CONTENT_WIDTH-26, 50)
    quote.drawOn(c, style.MARGIN+13, 228)
    style.tracking(c, 'CREATE  /  EDIT  /  RE-PERFORM  /  FINISH', style.MARGIN, 167, 6.8, .5, style.CYAN)
    style.tracking(c, 'EDITION I · MUSIC STUDIO v1', style.MARGIN, 70, 6.9, .9, style.MUTED)
    style.tracking(c, 'KAGGLE  /  BROWSER STUDIO  /  YOUR LIBRARY', style.MARGIN, 50, 6.4, .55, style.CYAN)
    c.showPage()
    metrics = []
    markdown = ['# Music Studio Field Guide', '', '*Edition I · 11 November 2026*', '', '[Open the designed PDF](../../Music-Studio-Guide.pdf)', '']
    for number, section in enumerate(data['pages'], 1):
        base(c, number+1, total)
        c.bookmarkPage(section['id'])
        c.addOutlineEntry(f'{number:02} {section["title"]}', section['id'], 1)
        style.tracking(c, f'SECTION  {number:02}', style.MARGIN, 680, 7.7, 2.2, style.CYAN, 'JJP-Bold')
        size = 25
        while style.pdfmetrics.stringWidth(section['title'], 'JJP-Bold', size) > style.CONTENT_WIDTH:
            size -= .5
        c.setFont('JJP-Bold', size)
        c.setFillColor(style.WHITE)
        c.drawString(style.MARGIN, 645, section['title'])
        style.tracking(c, section['kicker'], style.MARGIN, 625, 6.4, .1, style.MUTED)
        for scale in [1, .98, .96, .94, .92, .90, .88]:
            items = style.flowables(section['blocks'], scale)
            if style.measured(items) <= style.BODY_TOP-style.BODY_BOTTOM:
                break
        else:
            raise ValueError('Guide page overflows: ' + section['id'])
        y = style.BODY_TOP
        for item in items:
            _, h = item.wrap(style.CONTENT_WIDTH, style.HEIGHT)
            item.drawOn(c, style.MARGIN, y-h)
            y -= h
        metrics.append(dict(section=section['id'], page=number+1, scale=scale, body_bottom=round(y, 2), min_font=round(9*scale, 2)))
        c.showPage()
        markdown += [f'## {number:02} · {section["title"]}', '', section['kicker'], '']
        for block in section['blocks']:
            kind = block['type']
            if kind in ('text', 'lead'):
                markdown += [block['text'], '']
            elif kind in ('card', 'note'):
                markdown += ['**'+block['title']+'**', '', block['text'], '']
            elif kind in ('steps', 'bullets'):
                markdown += [f'{n}. {item}' for n, item in enumerate(block['items'], 1)] + ['']
            elif kind == 'links':
                markdown += [f'- [{v["label"]}]({v["url"]})' for v in block['items']] + ['']
            elif kind == 'table':
                markdown += ['| '+' | '.join(block['headers'])+' |', '| '+' | '.join('---' for _ in block['headers'])+' |']
                markdown += ['| '+' | '.join(row)+' |' for row in block['rows']] + ['']
    c.save()
    (ROOT / 'docs/music-studio/GUIDE.md').write_text('\n'.join(markdown))
    (ROOT / 'docs/music-studio/layout-check.json').write_text(json.dumps(dict(pages=total, layout=metrics), indent=2)+'\n')
    print(target.name, total, 'pages; minimum content bottom', min(p['body_bottom'] for p in metrics))


if __name__ == '__main__':
    build()
