# ScoreScribe

Sheet-music OCR that uses Claude to read the music and plain code to check it, then writes **MusicXML you open directly in MuseScore** (File → Open). It can also turn **a song recording into a lead sheet**: melody, lyrics and chord symbols (see *Songs from recordings*).

## Why it reads more cleanly than Audiveris

Audiveris recognises symbols one by one, then guesses how they fit together. When a stem, dot or flag is misread, the error flows straight into the file. ScoreScribe splits the work differently:

| Step | Who does it | Why |
|---|---|---|
| Find the staves, cut the page into systems, count the barlines | OpenCV (code) | Claude sees each system at nearly full scan resolution instead of a shrunken page |
| Read the notes, rhythms, clefs, dynamics, slurs, lyrics | **Claude** (vision) | Reads music the way a person does, using context. Each system is sent whole **plus two magnified halves**, so dots, flags and accidentals are about twice the size |
| Read it a second time, independently | **Claude** | Code compares the two readings note by note. Claude goes back only to the spots where they disagree and settles them |
| Apply key signature, carried accidentals and ties | Code | Claude only reports what is printed, so it never has to track accidentals in its head |
| Check every measure adds up, catch impossible octaves, compare the measure count with the barlines | Code | Bad measures go **back to Claude** with the exact problem ("staff 2, measure 3 adds up to 3.5 beats in 4/4") |
| Proofread | **Claude** | The transcription is engraved (Verovio), and Claude compares the engraving with the original side by side |
| Join measures cut in half at an image edge | Code | Screenshots and phone photos can split a measure between two images; the halves are put back together |
| Write MusicXML | music21 | Accidentals follow normal engraving rules, gaps in second voices are real gaps, and anything that failed a check is **red** |

The output is meant to look like a hand-made MuseScore file: no forced or redundant accidentals and no grey invisible rests. In MuseScore, **red** notes (only on the staff concerned) failed an automatic check. Measures Claude was unsure about are listed in `report.md` and on the results page. Add `--mark-uncertain` (or tick the box in the browser app) if you also want them coloured orange.

**Screenshots work.** You can upload several screenshots of one page, even if each one cuts the music off mid-measure; the pieces are joined in order. Put the images in reading order, as the app reads them by file name.

## Install

You need Python 3.10+ and an Anthropic API key from https://console.anthropic.com.

**Windows:** double-click `Install or Update (Windows).bat`, then `Start ScoreScribe.bat`. Run the install file again whenever you update ScoreScribe.

**Any system:**

```bash
cd scorescribe
pip install -e .
```

Set the key once (or paste it into the browser UI and tick "Remember"):

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # macOS / Linux
setx ANTHROPIC_API_KEY sk-ant-...          # Windows (open a new terminal afterwards)
```

Visual proofreading uses Verovio and resvg, which install from pip on Windows, macOS and Linux with nothing else to set up.

## Songs from recordings

Drop an mp3/wav/m4a/flac into the app (or run `scorescribe song-prepare song.mp3` / `scorescribe song song.mp3`). You get a lead sheet: the sung melody with lyrics and chord symbols, as MusicXML for MuseScore, plus `chords.txt`, a lyrics-with-chords chart.

**How it works.** Claude can't hear audio, so open-source tools do the listening and Claude does the musical thinking:

1. **Listening (on your computer or server):**
   - **Demucs** separates the voice from the band.
   - **librosa** finds the beats and measures the harmony on every beat (a 12-note pitch profile plus the bass note), then ranks candidate chords. It also tracks the sung pitch and cuts it into notes.
   - **Whisper** writes down the words with timestamps.
   - The results go into `analysis.md`.
2. **Pass 1 (Claude):** works out the bar lines, key and pickup. It chooses a chord progression that fits both the measurements and how harmony works, quantizes the melody, fixes pitch-tracking glitches and octave jumps, and puts the lyrics on the notes syllable by syllable.
3. **Check (code):** every chord is compared with the measured harmony and bass, and every melody note with the sung pitch. It also checks that rhythms add up, compares the lyrics with what Whisper heard, and lists out-of-key chords and melody notes that clash with the chords.
4. **Pass 2 (Claude), harmony review:** Claude goes through every flagged spot. Then it reviews the progression like a theorist: chord functions, cadences, a steady harmonic rhythm, repeated sections that match, and a melody that fits the chords. Anything still broken gets fixed before the file is built.

This works with an API key (Claude or Gemini, fully automatic) or with your **subscription**. Choose *My Claude Pro/Max plan* as the reader, open the folder in Claude Code, and say **"Make the lead sheet"**; it runs the same passes using `song-check` and `song-build`.

**Four-part harmony (SATB).** Tick *Four-part harmony* in the app (or add `--satb`) and Claude also arranges the song for choir in hymn style: soprano (the sung melody, with the lyrics), alto, tenor and bass, on two staves like a hymnal. The part-writing is checked by code:

- ranges, and rhythms that add up
- every note in its chord, the bass on the chord's bass note, and the third present in every chord
- no voice crossing, and no gaps that are too wide
- no parallel fifths or octaves, and no doubled leading tone

Claude then does a part-writing review pass. You get `song.musicxml` (the four-part score), `song_leadsheet.musicxml` (melody, lyrics and chords) and `chords.txt`. Chord symbols are left off the hymn score unless you tick *Also print chord symbols*. Shape notes work here too. This is an *arrangement* in hymn style built on the chords that were found, not a transcription of what a choir on the recording sang.

**Install (on the server):**

```bash
sudo apt install ffmpeg
source .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu   # CPU build, much smaller
pip install -e ".[audio]"
```

The first run downloads the Whisper (about 0.5 GB for `small`) and Demucs (about 80 MB) models. Separating a 4-minute song takes a few minutes on a CPU and wants about 4 GB of RAM. Untick "Separate the voice" in the app (or use `--no-separate`) to skip it on small servers; the melody comes out less reliable without it.

**For the best results:**
- Paste the lyrics in if you have them (the *Lyrics* box, or `--lyrics lyrics.txt`). Whisper mishears sung words, and the analysis can't recover a word it got wrong.
- One clear lead vocal works best. Choirs, heavy reverb, rap, or a singer buried in the mix make the melody harder to follow. Rubato (free tempo) confuses the beat grid.
- The chords are usually the strongest part. Always play the result back in MuseScore, especially the melody rhythms.

## Use it with your Claude or ChatGPT subscription (no API key)

Your regular plan can do the reading through the official coding agent that comes with it: **Claude Code** for Claude Pro/Max, or **Codex** for ChatGPT plans. ScoreScribe prepares the job, the agent reads the music, and ScoreScribe's tools check and build the result. It counts against your plan's normal usage limits; there is no separate bill.

1. In the browser app, choose **Reader → My Claude Pro/Max plan, via Claude Code** (or *My ChatGPT plan, via Codex*), drop in your score, and click **Prepare for my subscription**. (Command line: `scorescribe agent-prepare song.pdf`.)
2. Open the folder it shows:
   - **Claude:** Claude desktop app → **Code** tab → choose the folder. Or in a terminal, `cd` to it and run `claude`.
   - **ChatGPT:** in a terminal, `cd` to it and run `codex`.
3. Type **Transcribe this score**. The agent follows the `CLAUDE.md` / `AGENTS.md` instructions ScoreScribe wrote into the folder. It reads each line of music and runs `scorescribe agent-check .` (rhythm checks, measure counts, proof engravings to compare against the original). It fixes what the check finds, then runs `scorescribe rebuild` to write the MusicXML. Let it run those commands when it asks.
4. The finished file appears under **Recent runs → Download** in the app, and in that folder.

If you reach your plan's usage limit partway through, the lines of music already done are saved; say "continue" once your limit resets.

**Important for Claude:** if an `ANTHROPIC_API_KEY` environment variable is set on your computer, Claude Code uses that key (pay-per-use) instead of your subscription. The app warns you if it finds one. Remove it under Windows Settings → "Edit environment variables for your account".

This uses each agent exactly as it's meant to be used: you sign in to it yourself and ask it to do a task, and ScoreScribe never sees or handles your login. Sending large batches of scores through a subscription isn't what plans are sized for; for big jobs, an API key or Gemini is the better fit.

## Use it with a free Gemini key

Claude is the most accurate reader, but you can run ScoreScribe on Google Gemini instead, including its free tier:

1. Get a free key at https://aistudio.google.com/apikey.
2. In the browser app, choose **Reader → Gemini 3.8 Flash — free tier**, paste the key, and tick "Remember".
   On the command line: `set GEMINI_API_KEY=...` (Windows) or `export GEMINI_API_KEY=...`, then `scorescribe transcribe song.pdf --gemini`.

Things to know about the free tier:

- **Daily limits.** Google shows your current limits at https://aistudio.google.com/rate-limit (they're no longer published). With the default settings, each line of music takes about 3 to 5 requests, so a 3-page hymn is roughly 20 to 30. If a "per minute" limit is hit, ScoreScribe waits and carries on by itself. If the *daily* limit runs out, the run **pauses**: everything finished is saved. In the browser app, click **Resume** under *Recent runs* after the quota resets (midnight Pacific time). On the command line, run the same command again.
- **Fewer requests per page:** pick Accuracy → Standard (one reading instead of two), and/or untick Proofread. Each saves about a third.
- **You can switch reader when resuming.** For example, a run that paused because Claude credit ran out can be finished on Gemini, or the other way round. Choose the reader first, then click Resume.
- **Privacy:** Google may use free-tier requests to improve its products. The paid tier doesn't.
- **Accuracy:** the same checks (rhythm, barline count, two readings, proofreading) apply to every reader, but Gemini Flash may make more reading mistakes than Claude Opus on dense scores. Spot-check the result.

## Use it

### Browser (easiest)

```bash
scorescribe gui
```

Drop in a PDF or page photos. Click **Preview system split (free)** to check that every system was boxed correctly, then click **Transcribe** and download the MusicXML when it finishes.

### Command line

```bash
scorescribe transcribe song.pdf                       # writes song.musicxml
scorescribe transcribe page1.jpg page2.jpg -o out.musicxml --parts "Piano:2"
scorescribe transcribe choir.pdf --parts "Soprano,Alto,Tenor,Bass"
scorescribe segment song.pdf                          # free: just check the system split
```

Useful options:

- `--parts "Violin, Piano:2"`: give the layout if you know it (name:staves, top to bottom).
- `--staves-per-system 2`: force the grouping if the automatic split is wrong.
- `--model claude-sonnet-5-5`: cheaper and faster; Opus is the default because it is the most accurate.
- `--max-accuracy`: two readings plus extra thinking (`xhigh`). Use it for dense or difficult scores.
- `--fast`: one reading per system instead of two (cheaper, less accurate).
- `--no-zoom`: don't send the magnified halves (cheaper).
- `--no-proofread`: skip the engraving comparison.
- `--fresh`: ignore results cached from an earlier run (use this after updating, to get the new accuracy features on a score you already ran).
- `--workers 8`: read more systems in parallel.

### Shape notes (Aiken 7-shape or 4-shape)

MusicXML has shape-note heads built in, and MuseScore imports them as shape notes. Shapes follow movable do from the key signature, so minor keys are la-based as in shape-note hymnals, and key changes are followed.

```bash
scorescribe transcribe hymn.pdf --shape-notes aiken      # or 4shape (fa so la mi)
scorescribe rebuild hymn_work --shape-notes aiken         # re-output an earlier run, no API calls
scorescribe shapes anything.musicxml                      # convert ANY MusicXML file (e.g. exported from MuseScore)
```

In the browser app, use the **Noteheads** menu for new runs. The **Convert any MusicXML to shape notes** box at the bottom of the page converts a score you already have. To convert a MuseScore file, export it first (File → Export → MusicXML), convert it, then open the result.

Shapes are written on each note, so if you later change a note's pitch in MuseScore, you'll need to update its shape too (or convert again). Scanning a hymnal that is already printed in shape notes works too: Claude reads the notes by their position on the staff.

### Combine it with Audiveris

You can keep Audiveris in the loop as a first reader. Claude then corrects its draft against the image instead of starting from scratch:

```bash
scorescribe transcribe song.pdf --audiveris                 # runs Audiveris if it's installed
scorescribe transcribe song.pdf --draft song_audiveris.mxl  # or use a file you exported
```

The draft only lines up when Audiveris found the same number of systems; otherwise it is skipped with a message. Any OMR program that writes MusicXML with system breaks (oemer, homr, SmartScore and so on) works the same way.

### Fix things by hand, rebuild for free

Everything is cached in `<name>_work/`:

- `page01_systems.jpg`: where the systems were cut. Green lines mark the barlines that were found.
- `systems/sys001.png`, `sys001_zoom1.png` and `sys001_zoom2.png`: what Claude saw. `sys001.json` is what it wrote.
- `report.md`: every red or orange measure.

Re-running the same command reuses cached systems, so an interrupted run resumes where it stopped (`--fresh` starts over). You can edit any `sysNNN.json` (the notation is below) and rebuild without calling the API:

```bash
scorescribe rebuild song_work -o song.musicxml
```

## The notation Claude writes

Each voice is one line of tokens, for example `C5/4. D5/8 E5/4[staccato,mf] r/4`.

| Token | Meaning |
|---|---|
| `C4`, `F#5`, `Bb3`, `En4`, `F#!4` | Pitch. Notes covered by the key signature can be written with or without their sharp/flat; naturals against the key need `n`; `!` marks a printed courtesy accidental. Which accidentals get printed is decided by the program using normal engraving rules. |
| `/1 /2 /4 /8 /16 /32` and dots | Whole … 32nd; `C4/4.` is a dotted quarter |
| `C4+E4+G4/2` | Chord |
| `r/4`, `R`, `s/4` | Rest, whole-measure rest, invisible spacer |
| `A4/4~` | Tied to the next note (`C4~+E4/2` ties only the C) |
| `C5/8t3` | Triplet (`t5`, `t6`, `t3:2` …) |
| `g:D5/8`, `gs:D5/8` | Grace note / slashed grace note |
| `[...]` | staccato, accent, tenuto, marcato, fermata, trill, arpeggio, dynamics (`p`…`fff`, `sfz`), `slur_start`/`slur_end`, `cresc_start`/`dim_start`/`hairpin_end`, `8va`, `ped_start`/`ped_end`, `text=dolce`, `lyric=Glo-`, `fing=3` |

## Cost and speed

With the default settings, each system takes three to five Claude calls: two independent readings, one proofread, and one more each only if the readings disagree or a check fails. Follow-up calls reuse the images through prompt caching, so they cost much less than the first reading. A typical piano page has 4 to 6 systems, and a run usually finishes in a few minutes. The exact token usage is printed at the end of every run and written to `report.md`. To spend less, use `--fast`, `--no-zoom` or `--model claude-sonnet-5-5`. Try one page first to see what your scores cost.

## Limits worth knowing

- It handles printed music best. Neat handwriting often works; messy manuscripts are hit and miss.
- Cross-staff beaming comes out as notes on their own staff. Tremolos, ornaments beyond the common ones, and figured bass are not exported yet.
- Orchestral scores that hide empty staves work, but give `--parts` so Claude knows the full layout.
- A note slipped by an octave that still fits the rhythm can get past the automatic checks, which is exactly what the proofreading pass is for. Still listen through the result in MuseScore before you rely on it.

## Project layout and licences

```
scorescribe/
  imaging.py    page loading, deskew, staff detection (comb filter), barline detection, system cropping, zoom views
  prompts.py    instructions + tool schema for Claude
  pipeline.py   read twice → settle disagreements → check → repair → proofread → write; caching, parallelism
  notation.py   token parser; key and accidental logic
  model.py      assembles systems into a score and runs the checks
  build.py      music21 score and MusicXML export (colours, spanners, voltas)
  draft.py      MusicXML (Audiveris, etc.) → token drafts
  shapes.py     Aiken 7-shape / 4-shape noteheads for any MusicXML file
  render.py     Verovio engraving (rasterised with resvg) for proofreading
  gemini.py     Google Gemini reader (structured JSON output, rate-limit waits, daily-quota pause)
  agent.py      subscription mode: prepares CLAUDE.md/AGENTS.md jobs for Claude Code / Codex, and checks their work
  song/         recordings -> lead sheets: analyze.py (Demucs, librosa, Whisper), leadsheet.py (format, checks
                against the audio, MusicXML with chord symbols, chord chart), satb.py (four-part arrangement + part-writing
                checks), prompts.py, run.py (passes)
  web.py, cli.py
tests/          round-trip tests, a mocked pipeline, fake Claude/Gemini servers, a synthesized test song
```

ScoreScribe is MIT licensed. It builds on music21 (BSD), OpenCV (Apache 2.0), pypdfium2 (Apache 2.0/BSD), Verovio (LGPL 3.0), resvg-py (MIT/Apache 2.0), Flask (BSD), the Anthropic SDK (MIT), Google's google-genai SDK (Apache 2.0), and for recordings librosa (ISC), Demucs (MIT) and faster-whisper (MIT). Audiveris (AGPL) is only called as a separate program, never bundled.
