# Reusable prompt: topic-specific art style video

Copy everything in the block, fill the [BRACKETS], delete lines you don't need.

```text
Read the OpenMontage README, AGENT_GUIDE.md and the pipeline docs first, then make a [30]-second test video.
Work autonomously. Only ask me if something blocks you.

TOPIC
[Documentary-style video essay on <film / book / event / idea>.]
Thesis: "[one sentence that the whole video argues]"

FOOTAGE / RIGHTS RULE
Do NOT use clips, stills, posters, logos or likenesses from [the film / brand / person].
Use only free or open material as visual metaphors: [list of metaphors that fit the thesis].
Sources: Pexels, Pixabay, Unsplash, Archive.org, NASA, Wikimedia Commons.
Keep a list of every asset's source and licence in the project folder.
Before building, test which sources are reachable. If they are blocked, use a licensed fallback
(e.g. CC0 / public-domain images bundled in pip packages) or procedural art, and say so clearly.

UNIQUE ART STYLE: "[STYLE NAME]"   <- this is the most important section
Invent ONE original look that is the topic's own world, not a generic filter. Derive it from the thesis:
ask "what physical object, medium or era IS this story about?" and make the video look like that object.
Examples: a photocopied catalogue zine (consumerism), a redacted dossier (surveillance),
a hand-tinted glass slide (nostalgia), a VHS tracking-error tape (memory), a blueprint (architecture of control).
Define ALL of these in a reusable style config (palette, fonts, texture files, easing, transition list):
- Base treatment on every photo/clip: [e.g. high-contrast B&W halftone, crushed blacks, toner streaks]
- Surface/paper/screen: [material, damage, fasteners, tape, stains]
- Colour rule: [one accent colour, hex, where it may appear, where it must NEVER appear]
- Parody/diegetic graphics: [clean "official" graphics that get defaced/annotated as the narration turns]
- Typography: [rigid "official" face] + [rough "vandal"/hand face] + [cut-out / ransom style for key words]
- Motion language: [stepped 8-12 fps for graphics, smooth background, easing]
- Transitions (min 3, none default): [e.g. paper slap, copier flash, torn-page reveal, staple pop]
- Damage / glitch: [only at transitions, never constant]
- Grade: [desaturated, everything except the accent]
The style must carry the thesis: the way it gets damaged, defaced or corrected should mirror the argument.

FORMAT
1920x1080, 30 fps, about [30] seconds. Tone: [dry, controlled, a little cynical].
Narration: best available TTS; if no paid key, use a free local one and tell me which.
Word-level captions in the style's "vandal" typography, key words highlighted in the accent colour.
Subtle royalty-free music that drops out for one second before the last line.

SCRIPT (about [70] words, adjust to fit)
1. "..."
2. "..."
3. "..."
4. "..."
5. "..."

VISUAL TESTS (use all)
- Sourced footage: 4-6 real clips/photos found automatically, all given the style's base treatment.
- Cutouts: at least 2 subjects with background removed, with parallax and a hard paper-cut shadow.
- Motion graphics: [price tag + barcode drawing on, list items struck out one by one, etc.]
- Transitions: at least 3 different types from the style.
- Camera: slow pushes and drifts on every still.

PROCESS
1. Show a plan in under 200 words: pipeline, shot list, sources, how the style will be built. Then continue.
2. Build and render.
3. Watch your own output: extract frames at several points and check text overlap, bad crops, cutout edges,
   style consistency, caption timing and audio sync. Fix and re-render.
4. If something fails (missing key, no network, tool not installed), use a fallback and say so clearly.
5. Commit and push the project (never commit intermediate renders over 100 MB).

WHEN DONE, report in under 150 words:
- Path to the final MP4
- Pipeline and tools used
- Which visual tests passed, which used a fallback or failed
- Assets, sources, costs
- The single biggest limitation
```
