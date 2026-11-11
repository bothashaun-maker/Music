"""Small auditable changes to the immutable Linux Studio source checkout."""
from pathlib import Path
from .common import HERE


def apply(studio):
    studio = Path(studio)
    library = studio / 'crates/music-server/src/library.rs'
    old = 'let root=crate::studio_data_root().unwrap_or_else('
    new = 'let root=env::var_os("YUE_LIBRARY_ROOT").map(PathBuf::from).or_else(crate::studio_data_root).unwrap_or_else('
    text = library.read_text()
    if new not in text:
        if text.count(old) != 1:
            raise RuntimeError('Pinned library storage patch no longer applies.')
        library.write_text(text.replace(old, new))
    service = studio / 'crates/music-server/src/lib.rs'
    text = service.read_text()
    if 'mod kaggle_upload;' not in text:
        if text.count('mod mcp;') != 1:
            raise RuntimeError('Pinned upload module patch no longer applies.')
        text = text.replace('mod mcp;', 'mod mcp;\nmod kaggle_upload;')
        anchor = '.route("/health", get(health))'
        if text.count(anchor) != 1:
            raise RuntimeError('Pinned upload route patch no longer applies.')
        text = text.replace(anchor, anchor + '\n        .route("/kaggle/uploads", post(kaggle_upload::start))\n        .route("/kaggle/uploads/{id}", get(kaggle_upload::status).post(kaggle_upload::chunk))\n        .route("/kaggle/uploads/{id}/finish", post(kaggle_upload::finish))')
        service.write_text(text)
    (studio / 'crates/music-server/src/kaggle_upload.rs').write_text((HERE / 'kaggle_upload.rs').read_text())
    (studio / 'app/public/kaggle-upload.html').write_text((HERE / 'upload.html').read_text())
    sidebar = studio / 'app/components/Sidebar.tsx'
    text = sidebar.read_text()
    if 'Upload large WAV' not in text:
        lines = text.splitlines(True)
        for index, line in enumerate(lines):
            if "label={t('library')}" in line:
                lines.insert(index + 1, '          <NavItem icon={<Disc size={20} />} label="Upload large WAV" active={false} onClick={() => window.open("/kaggle-upload.html", "_blank")} isExpanded={isOpen} />\n')
                sidebar.write_text(''.join(lines))
                break
        else:
            raise RuntimeError('Pinned upload menu patch no longer applies.')
