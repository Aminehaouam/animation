# Asset sources & licences — XEROX CATALOGUE test episode (Fight Club essay)

**No footage, stills, posters, logos or likenesses from the film are used.**

## Why these sources
Pexels, Pixabay, Unsplash, Archive.org, NASA and Wikimedia Commons were all **blocked by this session's network proxy**
(HTTP 403 on CONNECT). Fallback: real public-domain / CC0 photographs that ship inside the `scikit-image` PyPI package
(data files, reached via allowed hosts), plus procedurally drawn graphics. None of these come from the requested sites.

## Photographs (real)
| File | Used as | Source | Licence |
|---|---|---|---|
| assets/src/coffee.png (+coffee_cut.png) | consumer object; cutout #1 | Pikolo Espresso Bar / Rachel Michetti, via scikit-image `data.coffee()` | CC0 |
| assets/src/coins.png | "collection / money" wallpaper | Brooklyn Museum Collection (Greek coins from Pompeii), via `data.coins()` | No known copyright restrictions |
| assets/src/brick.png | concrete / paving metaphor | CC0Textures (Bricks25), via `data.brick()` | CC0 |
| assets/src/gravel.png | concrete / rubble metaphor | CC0Textures (Gravel04), via `data.gravel()` | CC0 |
| assets/src/astronaut.png (+astronaut_cut.png) | "uniform" metaphor; cutout #2 (face cropped out, torso only) | NASA Great Images database, via `data.astronaut()` | Public domain (NASA) |
Background removal: `rembg` (u2net model, MIT / Apache-2.0).

## Procedural / original (no third-party material)
- Crowd silhouettes ("footage" panel, scene 3) and conveyor belt (scene 4): drawn per-frame in code — **not filmed footage**.
- Catalogue furniture line drawings, invented product names (BRÖNN, VÄLLA…), price tags, barcodes: drawn in code.
- All textures (paper, toner, lid shadow, burns, tape, staple): `textures.py`.
- Music + foley SFX: synthesised in `audio.py` (original, CC0).

## Voice
Kokoro-82M (Apache-2.0) via `kokoro-onnx`, voice `bm_george`, local/free. No API key, cost $0.

## Fonts
Inter (OFL), Bebas Neue (OFL), Humor Sans (OFL), Courier Prime (OFL), GNUTypewriter (GPL w/ font exception), DejaVu Serif (Bitstream Vera licence), Liberation Serif (OFL).

## Cost
$0.00 (no paid APIs).
