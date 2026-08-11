from __future__ import annotations

import json
from pathlib import Path

from scripts.extract_flickr_captions import extract_caption_records


def test_caption_extractor_is_stable_and_split_scoped(tmp_path: Path) -> None:
    source = tmp_path / "captions.json"
    source.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "imgid": 2,
                        "split": "test",
                        "sentences": [{"sentid": 3, "raw": " second "}],
                    },
                    {
                        "imgid": 1,
                        "split": "train",
                        "sentences": [{"sentid": 1, "raw": "ignored"}],
                    },
                    {
                        "imgid": 1,
                        "split": "test",
                        "sentences": [{"sentid": 0, "raw": "first"}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    records = extract_caption_records(source, split="test")
    assert records == [
        {"id": "flickr30k-1-0", "text": "first"},
        {"id": "flickr30k-2-3", "text": "second"},
    ]

