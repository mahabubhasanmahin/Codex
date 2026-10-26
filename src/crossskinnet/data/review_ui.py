"""Notebook buttons for explicit duplicate-review decisions; no automatic verdicts."""
from __future__ import annotations

import html
import io
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from crossskinnet.data.review import ReviewSession
from crossskinnet.data.dedup import _reviewed


def _thumbnail(path: str) -> bytes:
    """Render a readable local preview without altering the source image."""
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert('RGB')
        image.thumbnail((420,350))
        buffer = io.BytesIO()
        image.save(buffer,format='JPEG',quality=90)
    return buffer.getvalue()


def launch_review(config_path: Path = Path('configs/default.yaml')) -> Any:
    """Return an ipywidgets panel; decisions save directly to the shared review CSV."""
    import ipywidgets as widgets

    session = ReviewSession(config_path)
    if not session.rows:
        return widgets.HTML('No pending pairs in this prepared snapshot. Run metadata normalization and actual stage 04 before building/training.')
    index = 0
    heading, status = widgets.HTML(), widgets.HTML()
    left, right = widgets.Image(format='jpeg'), widgets.Image(format='jpeg')
    notes = widgets.Textarea(description='Notes',layout=widgets.Layout(width='95%',height='70px'))
    previous = widgets.Button(description='Previous')
    next_button = widgets.Button(description='Next')
    same = widgets.Button(description='Duplicate: True',button_style='warning')
    different = widgets.Button(description='Different: False',button_style='success')
    unsure = widgets.Button(description='Unsure: keep pending')

    def render() -> None:
        row = session.rows[index]
        decisions, records = _reviewed(session.reviews_path)
        marked = sum(decisions.get(identity,'') in ('true','false') for identity in session.ids)
        heading.value = (f'<b>Pair {index+1}/{len(session.rows)}</b> — distance {html.escape(row["distance"])}<br>'
            f'ID: {html.escape(row["pair_id"])}<br>'
            f'{html.escape(row["source_a"])} / {html.escape(row["image_id_a"])} &nbsp; ↔ &nbsp; '
            f'{html.escape(row["source_b"])} / {html.escape(row["image_id_b"])}')
        left.value, right.value = _thumbnail(row['image_path_a']),_thumbnail(row['image_path_b'])
        notes.value = records[row['pair_id']].get('notes','')
        status.value = (f'Marked {marked}/{len(session.rows)} in this snapshot. Current: '
            f'{html.escape(decisions.get(row["pair_id"],"") or "pending")}. '
            'This is not a training-readiness verdict; rerun preparation/stage 04 to refresh components.')
        previous.disabled, next_button.disabled = index==0,index==len(session.rows)-1

    def move(delta: int) -> None:
        nonlocal index
        index = max(0,min(len(session.rows)-1,index+delta))
        render()

    def save(decision: bool | None) -> None:
        session.save(session.rows[index]['pair_id'],decision,notes.value)
        move(1)

    previous.on_click(lambda button:move(-1))
    next_button.on_click(lambda button:move(1))
    same.on_click(lambda button:save(True))
    different.on_click(lambda button:save(False))
    unsure.on_click(lambda button:save(None))
    render()
    instruction = widgets.HTML('<b>Judge whether these are copies/transforms of the same image, not merely similar-looking lesions.</b> '
        'If uncertain, keep pending. Previous/Next do not save edited notes; a decision button saves.')
    return widgets.VBox([instruction,heading,widgets.HBox([left,right]),notes,
                         widgets.HBox([same,different,unsure]),widgets.HBox([previous,next_button]),status])
