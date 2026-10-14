#!/usr/bin/env python3
"""Writes crates/core/src/context/common_words.txt from a Qwen tokenizer's vocab.json.

    scripts/common-words.py <model dir>/tokenizer/vocab.json

A byte-level BPE vocabulary gives a whole token to each word frequent enough in its training text,
so a word that is one token with a leading space is common, and a recognizer spells it without
help. The list keeps those words, lowercase, two letters or longer. The Qwen tokenizer is
Apache-2.0.
"""
import json
import re
import sys
from pathlib import Path

vocab = json.loads(Path(sys.argv[1]).read_text())
words = sorted({t[1:].lower() for t in vocab if t.startswith("Ġ") and re.fullmatch("[A-Za-z]{2,}", t[1:])})
out = Path(__file__).resolve().parent.parent / "crates/core/src/context/common_words.txt"
out.write_text("\n".join(words) + "\n")
print(f"{len(words)} words in {out}")
