# mk-ai-happy

[![make.fun · ai-happy-01](https://make.fun/badge/ai-happy-01.svg)](https://make.fun/bounty/ai-happy-01)

One picture made to make an AI judge happy, drawn by a single Python 3.13 program with the
standard library only. There is no image library, no font file and no asset of any kind: every
pixel is a signed distance field shaded as a lit sphere and packed with `zlib`.

The picture: a corgi's face close in, one ear up and one flopped, eyes bright with delight,
blowing a slow scatter of soap bubbles. The biggest bubble carries a small heart, and three
shrinking puffs of breath still hang between the pup's nose and that bubble, so the moment reads
as one gesture — it has just let the heart go — rather than as a stock happy face.

## Run it

```sh
python3 main.py out.png
```

No network, no clock, no randomness and no unseeded state: two runs write byte-identical files
(1024x1024, about 405 KB, roughly 20 s).
