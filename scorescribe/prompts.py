"""Prompts and the tool schema Claude fills in for each system."""

SYSTEM_PROMPT = r"""You are an expert music engraver doing optical music recognition. You will be shown ONE system of printed (or neatly handwritten) sheet music at a time, cropped from a page, and you transcribe it exactly, measure by measure, by calling the `submit_transcription` tool.

Accuracy matters more than speed. Work staff by staff, measure by measure. For every note, find its staff position (line or space) relative to the clef, counting ledger lines carefully, then name the letter and octave. Check beams, flags, dots and rests so that every voice adds up to the time signature.

# Transcribe what is WRITTEN
- Give the letter and octave of the notehead position, in scientific pitch (middle C = C4; treble clef bottom line = E4, top line = F5; bass clef bottom line = G2, top line = A3; alto clef middle line = C4).
- Accidentals: `#` sharp, `b` flat, `n` natural, `##` double sharp, `bb` double flat.
  - A note with no accidental printed in front of it may be written either way: in D major, `F4` and `F#4` both mean F-sharp, because software applies the key signature, accidentals carried through the measure and ties. Software also decides which accidentals get printed, so writing `F#4` never adds a redundant sharp.
  - A printed NATURAL sign must always be written as `n` (`Fn4`). The same goes for any note that is natural where the key signature or an earlier accidental would alter it - otherwise it becomes sharp/flat.
  - Every accidental that changes the pitch must be included. A courtesy (cautionary) accidental - printed although the key signature or measure already gives that pitch, often in parentheses - is written with `!` after it: `F#!4`, `Bn!3`. Only use `!` for an accidental you can actually see printed.
- Shape-note heads (triangles, diamonds, squares, half-moons, as in shape-note hymnals) are read by their position on the staff exactly like round noteheads; the shape itself is not transcribed.
- Under an 8va / 8vb / 15ma line, still write the pitch as printed and tag each affected note with `[8va]` (or `[8vb]`, `[15ma]`).

# Token notation (one token per note, chord or rest; tokens separated by single spaces)
    PITCH/DURATION
- Durations: `1` whole, `2` half, `4` quarter, `8` eighth, `16`, `32`, `64`; add `.` per dot: `C5/4.` is a dotted quarter, `C5/8..` a double-dotted eighth.
- Chords: join pitches with `+`, lowest first: `C4+E4+G4/2`.
- Rests: `r/4`, `r/8.`. A whole-measure rest (centered whole rest) is just `R`.
- Invisible spacer (to fill a second voice where nothing is printed): `s/4`.
- Ties: put `~` after the duration of the first note: `A4/4~ A4/8`. In a chord where only some notes are tied, put `~` after those pitches: `C4~+E4+G4/2`. A note tied into the next system still gets `~`. Notes tied in from the previous system are written normally (no mark needed).
- Tuplets: add `t3` after the duration of every note in a triplet: `C5/8t3 D5/8t3 E5/8t3` (three eighths in the time of two). Quintuplets `t5`, sextuplets `t6`; explicit ratio `t3:2`. In compound meter a duplet is `t2`.
- Grace notes: prefix `g:` (or `gs:` for a slashed acciaccatura) and give the printed value: `gs:D5/8 C5/4`.
- Markings go in brackets after the token, comma separated, no spaces:
  articulations: staccato, staccatissimo, accent, tenuto, marcato, fermata, trill, mordent, turn, arpeggio, upbow, downbow, harmonic, breath
  dynamics (placed at that note): pppp ppp pp p mp mf f ff fff sf sfz sffz fp rfz fz
  spanners: slur_start, slur_end, cresc_start, dim_start, hairpin_end, ped_start, ped_end, 8va, 8vb, 15ma
  text: text=dolce (underscores for spaces: text=poco_rit.)
  lyrics: lyric=Glo- lyric=ri- lyric=a (verse 2: lyric2=word; end a syllable with - when the word continues)
  fingering: fing=3
  Example: `E5/8.[staccato,f,slur_start]`

# Voices
- Each staff has one or more voices. Voice 1 is the upper / main voice (stems up when two voices share a staff), voice 2 the lower. Every voice must fill the whole measure: use rests or `s` spacers so it adds up.
- Use a single voice with chords whenever the notes share one stem and one rhythm. Use a second voice only when rhythms really differ.

# Measure fields
- `staves`: one entry per staff shown in this system, with `staff` = the GLOBAL staff number (1 = top staff of the score; a piano is staves 1 and 2).
- In the first measure of the system give `clef` for every staff and `key` (number of sharps, negative for flats: -3 = three flats). Also give `clef` wherever a clef change appears mid-system, on the measure that starts in the new clef (a clef change in the middle of a measure: put it on the next measure).
- `time` only where a time signature is printed (e.g. "3/4", "6/8", "C" common time, "C|" cut time).
- `incomplete: true` for a pickup (anacrusis) or any measure that is deliberately shorter than the time signature.
- `barline` for the right-hand barline when it is not a plain single line: double, final, repeat_end, dashed. `repeat_start: true` when the measure begins with a start-repeat sign. `volta`: "1" or "2" for measures under a 1st/2nd ending bracket.
- `tempo`: {"text": "Allegro", "bpm": 120} where a tempo marking appears.
- Title and composer: report them if they are printed in the image.

# Cut-off images
The image may be a screenshot or photo in which a system is cut off at the left or right edge, so a measure is only partly visible. Transcribe exactly the part that is visible, and mark that measure `incomplete: true`. Software joins it with the rest of the measure from the next image. Do not invent the missing notes, and do not list a cut-off measure in `uncertain` just because it is cut off.

# Uncertainty
If something is genuinely illegible or ambiguous, still give your best reading and add an entry to `uncertain` naming the measure (1 = first measure in this image) and what is unclear. Never invent measures that are not in the image, and never skip any.

Before calling the tool, count the barlines so the number of measures is right, and check every voice's rhythm adds up."""


TOOL = {
    "name": "submit_transcription",
    "description": "Submit the complete transcription of every measure in the system image.",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Piece title if printed in this image"},
            "composer": {"type": "string"},
            "parts": {
                "type": "array",
                "description": "Instruments/voices top to bottom with how many staves each uses (piano = 2). Required on the first system.",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "staves": {"type": "integer"}},
                    "required": ["name", "staves"],
                },
            },
            "measures": {
                "type": "array",
                "description": "Every measure in the image, left to right.",
                "items": {
                    "type": "object",
                    "properties": {
                        "time": {"type": "string"},
                        "key": {"type": "integer"},
                        "incomplete": {"type": "boolean"},
                        "barline": {"type": "string", "enum": ["regular", "double", "final", "repeat_end", "dashed"]},
                        "repeat_start": {"type": "boolean"},
                        "repeat_times": {"type": "integer"},
                        "volta": {"type": "string"},
                        "tempo": {"type": "object", "properties": {"text": {"type": "string"}, "bpm": {"type": "number"}}},
                        "staves": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "staff": {"type": "integer"},
                                    "clef": {"type": "string", "description": "G2 treble, F4 bass, C3 alto, C4 tenor, G2-8 tenor-voice treble, perc"},
                                    "voices": {"type": "array", "items": {"type": "string"}, "description": "One token string per voice"},
                                    "texts": {"type": "array", "items": {"type": "string"}, "description": "Other text at the start of this measure (rehearsal marks, instructions)"},
                                },
                                "required": ["staff", "voices"],
                            },
                        },
                    },
                    "required": ["staves"],
                },
            },
            "uncertain": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"measure": {"type": "integer"}, "staff": {"type": "integer"}, "note": {"type": "string"}},
                    "required": ["measure", "note"],
                },
            },
            "changes": {"type": "string", "description": "When resubmitting: what you changed and why"},
        },
        "required": ["measures"],
    },
}


def first_message(index: int, total: int, n_staves_detected: int, parts: list[dict] | None,
                  hint: dict | None) -> str:
    hint = hint or {}
    lines = []
    if hint.get("zooms"):
        lines.append(f"Image 1 is the whole system. Images 2 to {1 + hint['zooms']} are magnified views of its left and "
                     "right parts (they overlap a little) so you can see small details: accidentals, dots, flags, beams, "
                     "ledger lines. Transcribe the system ONCE, measure by measure across the whole system, using image 1 "
                     "for the overall layout and the magnified views to read each symbol.")
    lines.append(f"This is system {index} of {total} in the score.")
    if n_staves_detected:
        lines.append(f"Staff-line detection found {n_staves_detected} staves in this image.")
    if hint.get("measures"):
        lines.append(f"Barline detection counted {hint['measures']} measures in this system (usually right, but long "
                     "stems or repeat signs can fool it - the image is the authority).")
    if parts:
        total_staves = sum(p['staves'] for p in parts)
        desc = "; ".join(f"{p['name']}: {p['staves']} staff" + ("s" if p['staves'] > 1 else "") for p in parts)
        lines.append(f"The score layout is: {desc} ({total_staves} staves in all, numbered 1 to {total_staves} from the top).")
        if n_staves_detected and n_staves_detected != total_staves:
            lines.append("This system shows a different number of staves than the full layout, so some may be hidden here; "
                         "identify each staff by its label/clef/range and use its global staff number.")
    else:
        lines.append("This is the first system: also fill in `parts` (instrument names and staves per instrument, top to bottom).")
    if hint.get("time"):
        lines.append(f"Time signature in effect before this system (from earlier systems): {hint['time']}. "
                     "Only fill `time` if a time signature is printed in this image.")
    if hint.get("draft"):
        lines.append("\nAnother OMR program produced the draft below for this system, in the same JSON measure format. "
                     "Drafts like this usually contain mistakes: wrong pitches or octaves, missed or invented accidentals, "
                     "missing or extra notes, wrong rhythms, merged or lost voices, misread clefs. Use it only as a starting "
                     "point; the image is the authority. Check every measure against the image.\n"
                     "<draft>\n" + hint["draft"] + "\n</draft>")
    lines.append("Transcribe every measure in the image and call submit_transcription.")
    return "\n".join(lines)


def recount_message(detected: int, given: int) -> str:
    return (f"Barline detection counted {detected} measures in this system, but your transcription has {given}. "
            "The detector can be fooled (long stems, repeat signs, a barline hidden by a slur), so it may be wrong. "
            "Count the barlines in the image again carefully, measure by measure.")


def consensus_message(diffs: list[str], other_json: str | None) -> str:
    out = ["A second, independent reading of this same system disagreed with yours in these places "
           "(\"yours\" is your transcription above, \"other\" is the independent reading):"]
    out += diffs
    if other_json:
        out.append("\nThe other reading in full:\n" + other_json)
    out.append("\nFor each disagreement, look closely at that spot in the images (use the magnified views) and decide "
               "which is right; either reading may be the wrong one, and sometimes neither is. Then call "
               "submit_transcription with the COMPLETE corrected system. If you cannot decide a spot, choose the more "
               "likely reading and list it in `uncertain`. Summarise your decisions in `changes`.")
    return "\n".join(out)


def repair_message(errors: list[str], warnings: list[str], recount: str | None = None) -> str:
    out = ["Automatic checks found problems in your transcription of this system:"]
    out += [f"- ERROR {e}" for e in errors]
    if recount:
        out.append(f"- MEASURE COUNT: {recount}")
    if warnings:
        out.append("Things to double-check (these may be fine as they are):")
        out += [f"- {w}" for w in warnings]
    out.append("\nLook at the images again closely for those measures: re-count beams, flags, dots, rests and tuplets, "
               "and re-read the pitches. Then call submit_transcription again with the COMPLETE corrected system "
               "(all measures, not only the changed ones) and summarise what you changed in `changes`. If a check is "
               "about something that really is printed that way, keep it and say so in `changes`.")
    return "\n".join(out)


PROOFREAD_MESSAGE = (
    "Proofreading pass. The image above is an engraving of exactly what you transcribed (measure numbers shown "
    "count from 1 within this system; accidentals are shown as the software resolved them from the key signature). "
    "Compare it with the original image measure by measure and staff by staff: pitches and octaves, accidentals, "
    "rhythms, rests, ties, voices and the number of measures. Ignore differences in layout, spacing, stem direction, "
    "beaming style and fonts.\n"
    "Call submit_transcription with the complete corrected system. If everything already matches, resubmit it "
    "unchanged and set `changes` to \"none\"."
)

PROOFREAD_TEXT_MESSAGE = (
    "Proofreading pass. Re-examine the original image measure by measure and staff by staff and compare it with "
    "your transcription above: pitches and octaves (count ledger lines again), printed accidentals, rhythms, rests, "
    "ties and the number of measures. Call submit_transcription with the complete corrected system. If everything "
    "already matches, resubmit it unchanged and set `changes` to \"none\"."
)
