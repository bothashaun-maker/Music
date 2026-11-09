"""Build the JJP music companion PDF and editable Markdown from guide.json.

Requires ReportLab. Fonts: DejaVu Sans, installed on the document-build machine.
This script creates documentation only; it never imports a music runtime/model.
"""
from pathlib import Path
import html
import json
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/music-production-system/guide.json'
WIDTH, HEIGHT = 540, 765.12
MARGIN, BODY_TOP, BODY_BOTTOM = 48, 608, 70
CONTENT_WIDTH = WIDTH - 2 * MARGIN
BG = colors.HexColor('#0d0e10')
PANEL = colors.HexColor('#151618')
WHITE = colors.HexColor('#eee8e4')
BODY = colors.HexColor('#c9c5c2')
MUTED = colors.HexColor('#969393')
CORAL = colors.HexColor('#d98082')
CYAN = colors.HexColor('#76dcdd')
LINE = colors.HexColor('#333638')


def fonts():
    folder = Path('/usr/share/fonts/truetype/dejavu')
    for name, filename in [('JJP', 'DejaVuSans.ttf'), ('JJP-Bold', 'DejaVuSans-Bold.ttf'),
                           ('JJP-Italic', 'DejaVuSans-Oblique.ttf'),
                           ('JJP-BoldItalic', 'DejaVuSans-BoldOblique.ttf'),
                           ('JJP-Mono', 'DejaVuSansMono.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(folder / filename)))
    pdfmetrics.registerFontFamily('JJP', normal='JJP', bold='JJP-Bold',
                                italic='JJP-Italic', boldItalic='JJP-BoldItalic')


def tracking(c, text, x, y, size=7.5, spacing=1.5, color=MUTED, font='JJP'):
    c.saveState()
    obj = c.beginText(x, y)
    obj.setFont(font, size)
    obj.setFillColor(color)
    obj.setCharSpace(spacing)
    obj.textOut(text)
    c.drawText(obj)
    c.restoreState()


def safe(text):
    return html.escape(text).replace('\n', '<br/>')


def styles(scale):
    return {
        'body': ParagraphStyle('body', fontName='JJP', fontSize=9.7*scale,
                               leading=14.7*scale, textColor=BODY, splitLongWords=True),
        'lead': ParagraphStyle('lead', fontName='JJP', fontSize=10.6*scale,
                               leading=16.1*scale, textColor=WHITE),
        'cell': ParagraphStyle('cell', fontName='JJP', fontSize=9.0*scale,
                               leading=13.0*scale, textColor=BODY, splitLongWords=True),
        'label': ParagraphStyle('label', fontName='JJP-Bold', fontSize=7.3*scale,
                                leading=11*scale, textColor=CYAN),
        'link': ParagraphStyle('link', fontName='JJP-Bold', fontSize=8.1*scale,
                               leading=12*scale, textColor=CYAN),
        'number': ParagraphStyle('number', fontName='JJP-Bold', fontSize=10.1*scale,
                                 leading=14.7*scale, textColor=CYAN),
    }


def flowables(blocks, scale):
    st = styles(scale)
    result = []
    for block in blocks:
        kind = block['type']
        if kind in ('text', 'lead'):
            result += [Paragraph(safe(block['text']), st['lead' if kind == 'lead' else 'body']), Spacer(1, 9*scale)]
        elif kind in ('card', 'note'):
            border = CORAL if kind == 'note' else CYAN
            table = Table([[Paragraph(safe(block['title'].upper()), st['label'])],
                           [Paragraph(safe(block['text']), st['body'])]],
                          colWidths=[CONTENT_WIDTH])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), PANEL),
                ('LINEBEFORE', (0, 0), (0, -1), 1.5, border),
                ('TOPPADDING', (0, 0), (-1, 0), 10*scale),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 3*scale),
                ('TOPPADDING', (0, 1), (-1, 1), 0),
                ('BOTTOMPADDING', (0, 1), (-1, 1), 10*scale),
                ('LEFTPADDING', (0, 0), (-1, -1), 12*scale),
                ('RIGHTPADDING', (0, 0), (-1, -1), 12*scale),
            ]))
            result += [table, Spacer(1, 11*scale)]
        elif kind in ('steps', 'bullets'):
            rows = []
            for number, item in enumerate(block['items'], 1):
                label = f'{number:02}' if kind == 'steps' else '•'
                rows.append([Paragraph(label, st['number']), Paragraph(safe(item), st['body'])])
            table = Table(rows, colWidths=[25*scale, CONTENT_WIDTH-25*scale])
            table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (0, -1), 0),
                ('RIGHTPADDING', (0, 0), (0, -1), 5*scale),
                ('LEFTPADDING', (1, 0), (1, -1), 0),
                ('RIGHTPADDING', (1, 0), (1, -1), 0),
                ('TOPPADDING', (0, 0), (-1, -1), 0),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 7*scale),
            ]))
            result += [table, Spacer(1, 3*scale)]
        elif kind == 'table':
            fractions = block.get('widths') or [1/len(block['headers'])]*len(block['headers'])
            rows = [[Paragraph(safe(t), st['label']) for t in block['headers']]]
            rows += [[Paragraph(safe(value), st['cell']) for value in row] for row in block['rows']]
            table = Table(rows, colWidths=[CONTENT_WIDTH*f for f in fractions])
            rules = [
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('BACKGROUND', (0, 0), (-1, 0), PANEL),
                ('LEFTPADDING', (0, 0), (-1, -1), 9*scale),
                ('RIGHTPADDING', (0, 0), (-1, -1), 9*scale),
                ('TOPPADDING', (0, 0), (-1, -1), 7*scale),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 7*scale),
                ('LINEBELOW', (0, 0), (-1, 0), .6, CORAL),
            ]
            for i in range(1, len(rows)):
                rules.append(('LINEBELOW', (0, i), (-1, i), .35, LINE))
            table.setStyle(TableStyle(rules))
            result += [table, Spacer(1, 11*scale)]
        elif kind == 'links':
            for item in block['items']:
                url = html.escape(item['url'], quote=True)
                result += [Paragraph(f'<link href="{url}" color="#76dcdd"><u>{safe(item["label"])}</u></link>', st['link']), Spacer(1, 6*scale)]
            result += [Spacer(1, 3*scale)]
        else:
            raise ValueError('Unsupported guide block: ' + kind)
    return result


def measured(items):
    return sum(item.wrap(CONTENT_WIDTH, HEIGHT)[1] for item in items)


def base_page(c, page, total):
    c.setFillColor(BG)
    c.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
    c.setStrokeColor(CORAL)
    c.setLineWidth(.9)
    for x, y, sx, sy in [(18, 18, 1, 1), (WIDTH-18, 18, -1, 1),
                          (18, HEIGHT-18, 1, -1), (WIDTH-18, HEIGHT-18, -1, -1)]:
        c.line(x, y, x+13*sx, y)
        c.line(x, y, x, y+13*sy)
    if page > 1:
        tracking(c, 'JESUSJUICEPRODUCTIONS', MARGIN, HEIGHT-39, 6.4, 1.15, CYAN)
        c.setFont('JJP', 6.8)
        c.setFillColor(MUTED)
        c.drawRightString(WIDTH-MARGIN, HEIGHT-39, 'STUDIO OPERATIONS  /  EDITION I')
        c.setStrokeColor(LINE)
        c.line(MARGIN, 48, WIDTH-MARGIN, 48)
        c.setFont('JJP', 7)
        c.drawString(MARGIN, 33, 'MUSIC PRODUCTION SYSTEM  ·  9 NOV 2026')
        c.drawRightString(WIDTH-MARGIN, 33, f'{page:02} / {total:02}')
        c.setFillColor(CYAN)
        c.drawRightString(WIDTH-MARGIN, HEIGHT-56, 'BACK TO LAUNCH DESK')
        c.linkRect('', 'desk', (WIDTH-MARGIN-115, HEIGHT-60, WIDTH-MARGIN, HEIGHT-47), relative=0, thickness=0)


def cover(c, data, total):
    base_page(c, 1, total)
    c.bookmarkPage('cover')
    c.addOutlineEntry('JJP Music Production System', 'cover', 0)
    c.setStrokeColor(colors.HexColor('#385455'))
    c.roundRect(MARGIN, 584, 296, 24, 2, stroke=1, fill=0)
    tracking(c, data['brand'], MARGIN+12, 592, 7.1, 2, CYAN)
    for text, y, color in [('THE MUSIC', 523, WHITE), ('PRODUCTION', 471, WHITE), ('SYSTEM', 419, CORAL)]:
        c.setFillColor(color)
        c.setFont('JJP-Bold', 43)
        c.drawString(MARGIN, y, text)
    subtitle = Paragraph(safe(data['subtitle']), ParagraphStyle('cover-sub', fontName='JJP',
                         fontSize=11.1, leading=18.5, textColor=MUTED))
    _, h = subtitle.wrap(CONTENT_WIDTH, 100)
    subtitle.drawOn(c, MARGIN, 370-h)
    c.setFillColor(PANEL)
    c.roundRect(MARGIN, 205, CONTENT_WIDTH, 72, 3, fill=1, stroke=0)
    quote = Paragraph('“Make a keeper. Save the work.<br/>Come back with a plan.”',
                      ParagraphStyle('quote', fontName='JJP-Italic', fontSize=12.4,
                                     leading=19, textColor=WHITE))
    quote.wrap(CONTENT_WIDTH-26, 50)
    quote.drawOn(c, MARGIN+13, 228)
    x = MARGIN
    for label in ['CREATE / COVER', 'STEMS', 'TIMESTAMPS', 'REAPER + RELEASE']:
        w = pdfmetrics.stringWidth(label, 'JJP', 6.8) + 17
        c.setStrokeColor(LINE)
        c.roundRect(x, 160, w, 20, 2, stroke=1, fill=0)
        tracking(c, label, x+8, 167, 6.8, 0, MUTED)
        x += w+8
    tracking(c, data['edition'], MARGIN, 70, 6.9, .9, MUTED)
    tracking(c, 'KAGGLE  /  COMFYUI  /  REAPER', MARGIN, 50, 6.9, .9, CYAN)
    c.setFont('JJP', 7)
    c.setFillColor(MUTED)
    c.drawRightString(WIDTH-MARGIN, 50, '9 NOV 2026')
    c.showPage()


def build_pdf(data):
    fonts()
    target = ROOT / 'JJP_Music_Production_System.pdf'
    c = canvas.Canvas(str(target), pagesize=(WIDTH, HEIGHT), pageCompression=1, invariant=1)
    c.setTitle('JesusJuiceProductions — The Music Production System · Edition I')
    c.setAuthor('JesusJuiceProductions')
    c.setSubject('Practical companion: song creation, covers, model storage, stems, REAPER, masters, timestamps and delivery')
    total = len(data['pages'])+1
    cover(c, data, total)
    metrics = []
    for number, section in enumerate(data['pages'], 1):
        base_page(c, number+1, total)
        c.bookmarkPage(section['id'])
        c.addOutlineEntry(f'{number:02}  {section["title"]}', section['id'], 1)
        tracking(c, f'SECTION  {number:02}', MARGIN, 680, 7.7, 2.2, CYAN, 'JJP-Bold')
        size = 25
        while pdfmetrics.stringWidth(section['title'], 'JJP-Bold', size) > CONTENT_WIDTH:
            size -= .5
        c.setFillColor(WHITE)
        c.setFont('JJP-Bold', size)
        c.drawString(MARGIN, 645, section['title'])
        tracking(c, section['kicker'], MARGIN, 625, 6.7, .3, MUTED)
        for scale in [1, .98, .96, .94, .92, .90, .88]:
            items = flowables(section['blocks'], scale)
            height = measured(items)
            if height <= BODY_TOP-BODY_BOTTOM:
                break
        else:
            raise ValueError(f'Section {section["id"]} overflows: {height:.1f} pt')
        y = BODY_TOP
        for item in items:
            _, h = item.wrap(CONTENT_WIDTH, HEIGHT)
            item.drawOn(c, MARGIN, y-h)
            y -= h
        metrics.append({'section': section['id'], 'page': number+1, 'scale': scale,
                        'body_bottom': round(y, 2), 'min_body_font': round(9.0*scale, 2)})
        c.showPage()
    c.save()
    return target, metrics


def markdown(data):
    out = ['# JJP Music Production System', '', '*Edition I · Studio operations · 9 November 2026*', '',
           '[Open the designed PDF](JJP_Music_Production_System.pdf) · [One-page launch desk](MUSIC_LAUNCH_DESK.md)', '']
    for number, section in enumerate(data['pages'], 1):
        out += [f'## {number:02} · {section["title"]}', '', f'*{section["kicker"]}*', '']
        for block in section['blocks']:
            kind = block['type']
            if kind in ('text', 'lead'):
                out += [block['text'], '']
            elif kind in ('card', 'note'):
                out += ['**'+block['title']+'**', '']
                if block['title'] == 'LOG LOCATIONS':
                    out += ['```text', block['text'], '```', '']
                else:
                    out += [block['text'].replace('\n', '<br>\n'), '']
            elif kind in ('steps', 'bullets'):
                out += [(f'{n}. ' if kind == 'steps' else '- ') + item
                        for n, item in enumerate(block['items'], 1)] + ['']
            elif kind == 'links':
                out += [f'- [{item["label"]}]({item["url"]})' for item in block['items']] + ['']
            elif kind == 'table':
                out += ['| '+' | '.join(block['headers'])+' |',
                        '| '+' | '.join('---' for _ in block['headers'])+' |']
                out += ['| '+' | '.join(value.replace('|', '\\|') for value in row)+' |' for row in block['rows']] + ['']
    target = ROOT / 'MUSIC_PRODUCTION_SYSTEM.md'
    target.write_text('\n'.join(out))
    return target


def launch_desk(data):
    links = data['links']
    text = f'''# JJP Music Launch Desk

[Full production-system PDF](JJP_Music_Production_System.pdf) · [Editable full guide](MUSIC_PRODUCTION_SYSTEM.md)

Choose the job you need. Already have a finished song? Skip generation.

| Your job | Open this | Kaggle setting |
| --- | --- | --- |
| Create a song | [Music Creation]({links['create']}) → Full Song | T4 ×2 · Internet ON · existing YuE2 assets |
| Cover / restyle | [Same notebook]({links['create']}) → Cover / Restyle | T4 ×2 · Internet ON · existing YuE2 assets |
| Separate stems | [Stem Studio]({links['stems']}) | T4 ×2 · Internet ON · saved stem Dataset |
| Time lyrics / subtitles | [Timestamp Studio]({links['timestamp']}) | T4 ×2 recommended · Internet ON |
| Repair markers locally | [REAPER shift action]({links['reaper']}) | REAPER · no Kaggle run |
| Artwork | [Image Studio]({links['images']}) | Its own model packs/settings |
| Optional movie visuals | [Movie workflow]({links['movies']}) | Its own model packs/settings |

## Every session

1. Attach the saved inputs. Select the appropriate GPU setting.
2. Run the notebook's single startup code cell. Wait for its ready message and links.
3. Open the new Cloudflare canvas link. Keep the startup cell running.
4. Select the source and settings. Press Run once; listen and review.
5. Download the result ZIP or save a notebook version with output files before closing.

`FutureWarning` is not an error. An old Cloudflare link is not a permanent studio address.
Stop the previous startup cell before switching music tools in the same runtime.

## Model storage — first time only

- [Music asset builder]({links['music_assets']}): reuse your existing `yue2-music-assets`; no rebuild for a new song.
- [Stem model Dataset builder]({links['stem_assets']}): **Accelerator None**, Internet ON. Named Dataset: `janpierrrewiet/stem-studio-assets-v1-f6815b582a` (about 2.1 GiB).
- Timestamp models/frame caches can be reused from saved Studio outputs. There is no equivalent automatic named-Dataset builder in the current Timestamp notebook.
- [Image model packs]({links['image_assets']}) and [upscaler model packs]({links['upscale_assets']}) are separate from music assets.

## Keep the order clear

New song or existing recording → optional stems/remix → **approved final audio** → timestamps → reviewed subtitles/markers → visuals/release pack.

Keep original WAVs and manual REAPER corrections. The model Dataset does not preserve generated songs.

[Song brief template](docs/music-production-system/templates/song_brief.md) · [Cue review sheet](docs/music-production-system/templates/cue_review.csv) · [Production log](docs/music-production-system/templates/production_log.csv)
'''
    (ROOT / 'MUSIC_LAUNCH_DESK.md').write_text(text)


def main():
    data = json.loads(DATA.read_text())
    pdf, metrics = build_pdf(data)
    markdown(data)
    launch_desk(data)
    artifact = ROOT / 'docs/music-production-system/layout-check.json'
    artifact.write_text(json.dumps({'pages': len(data['pages'])+1, 'layout': metrics}, indent=2)+'\n')
    print(pdf)
    print('Pages:', len(data['pages'])+1, 'Minimum content bottom:', min(m['body_bottom'] for m in metrics))


if __name__ == '__main__':
    main()
