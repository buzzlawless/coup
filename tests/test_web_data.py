"""Tests for the data the browser explorer reads."""

from __future__ import annotations

import gzip
import itertools
import json
from collections import Counter

import pytest

from analysis.build_web_data import MAX_COINS, OUT, SHORT, slug
from coup import Card

MANIFEST = OUT / "manifest.json"

pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(), reason="run `python -m analysis.build_web_data`"
)


def legal_pairs():
    return list(itertools.combinations_with_replacement(list(Card), 2))


def test_every_pair_of_upcards_has_a_shard():
    """The page picks a file from this manifest, so a gap here is a dead UI."""
    manifest = json.loads(MANIFEST.read_text())
    listed = {tuple(sorted(entry["dead"])) for entry in manifest}
    expected = {tuple(sorted(SHORT[str(c)] for c in pair)) for pair in legal_pairs()}
    assert listed == expected
    assert len(expected) == 15


def test_every_shard_named_in_the_manifest_exists():
    for entry in json.loads(MANIFEST.read_text()):
        assert (OUT / entry["file"]).exists(), entry["file"]


def test_the_page_never_rebuilds_a_file_name():
    """The generator names the files; the page must ask rather than guess.

    A plain alphabetical sort in the page and a card-order sort in the
    generator disagree on six of the fifteen pairs, which is exactly the bug
    resolving names through the manifest removes.
    """
    app = (OUT.parent / "app.js").read_text()
    assert "manifest.json" in app
    assert '.sort().join("-")' not in app


@pytest.mark.parametrize("pair", [(Card.AMBASSADOR, Card.CAPTAIN), (Card.DUKE, Card.DUKE)])
def test_a_shard_covers_every_start_for_its_upcards(pair):
    data = json.loads(gzip.decompress((OUT / f"{slug(pair)}.json.gz").read_bytes()))
    dead = Counter(SHORT[str(c)] for c in pair)
    expected = {
        f"{SHORT[str(a)]}{SHORT[str(b)]}{x}.{y}.0"
        for a, b in itertools.product(list(Card), repeat=2)
        if all(n <= 3 for n in (Counter([SHORT[str(a)], SHORT[str(b)]]) + dead).values())
        for x in range(MAX_COINS + 1)
        for y in range(MAX_COINS + 1)
    }
    assert set(data["starts"]) == expected
