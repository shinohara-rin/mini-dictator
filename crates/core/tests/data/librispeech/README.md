# LibriSpeech samples

Four utterances from the LibriSpeech ASR corpus `test-clean` set
(<https://www.openslr.org/12>), two speakers of each sex, for the end-to-end test in
`crates/core/tests/e2e.rs`. Converted from FLAC to 16 kHz mono 16-bit WAV with ffmpeg; the audio
is otherwise unchanged. `transcripts.txt` holds the corpus transcripts.

| File | Speaker | Reader |
| --- | --- | --- |
| `1089-134686-0000.wav` | 1089, M | Peter Bobbe |
| `1284-1180-0001.wav` | 1284, F | Daniel Anaya |
| `237-126133-0003.wav` | 237, F | rachelellen |
| `260-123286-0002.wav` | 260, M | Brad Bush |

LibriSpeech (c) 2014 by Vassil Panayotov, licensed under the Creative Commons Attribution 4.0
International License (<http://creativecommons.org/licenses/by/4.0/>). It is derived from
LibriVox public domain audiobooks read by the volunteers named above.
