"""Instructions for turning a recording's analysis into a lead sheet, and for the review passes."""

from .. import prompts as scan_prompts

_TOKENS = scan_prompts.SYSTEM_PROMPT[scan_prompts.SYSTEM_PROMPT.index("# Token notation"):
                                     scan_prompts.SYSTEM_PROMPT.index("# Voices")]

GUIDE = r"""You are an expert music transcriber and arranger making a LEAD SHEET of a recorded song: the sung melody with its lyrics, plus chord symbols. You cannot hear the recording. Audio software has measured it for you; the measurements are in the analysis below (beat by beat). Your job is to turn imperfect measurements into the most musically plausible, accurate lead sheet, using your knowledge of how songs, melodies and harmony work.

# Reading the analysis
- Beats: the beat tracker's beats are numbered 0, 1, 2, ... with their time. Every position in the analysis is given in these beat numbers (e.g. 12.50 = halfway through beat 12).
- accent: how strong the attack on that beat is (0-1). Downbeats (beat 1 of a bar) are often, not always, stronger.
- bass: the lowest strong pitch class on that beat. Chord candidates: chords whose notes best match the measured harmony, with a fit score; "smoothed" is the most likely chord sequence. Strongest pitch classes: what's sounding most.
- Sung notes: pitch @ position within the beat ~ length in beats; "?" = low confidence. Pitch tracking can jump an octave, produce short glitches at note edges (scoops, slides, breaths), and merge two syllables sung on the same pitch.
- Words heard: from speech recognition, with beat positions. It mishears words, especially when sung; use rhyme, grammar and meaning to correct them. If the user supplied lyrics, those are correct: use them.

# Steps
1. BEAT GRID. Decide the meter and where bar lines go. Bars usually begin where chords change, where phrases start, where accents are strong and on stressed syllables. Set `downbeat_beat` to the analysis beat number of beat 1 of the first FULL bar. If singing starts before that, the first measure is a pickup: give it `"incomplete": true` with just the pickup notes. The tracker may count at half or double speed: set `beat_ql` (quarter notes per tracked beat; normally 1.0). Use a time signature that fits (4/4, 3/4, 6/8 = beat_ql 1.5 when the tracker follows dotted quarters, 12/8...).
2. KEY. Use the key estimates, the chords and the melody's final notes. `key` = number of sharps (negative = flats); `mode` major/minor.
3. HARMONY. Choose the chord for each stretch of the song.
   - The bass note and the strongest pitch classes are the best evidence; the candidate list is a guide, not an answer.
   - Chords normally change on beat 1, sometimes on beat 3 (or other beats in a clearly faster harmonic rhythm). Don't put a new chord on every beat just because melody notes or passing tones move.
   - Make it make sense musically: diatonic chords of the key are the most likely; dominant chords lead to their tonic; phrases end with cadences (V-I, IV-I, ii-V-I); a borrowed or secondary chord (e.g. V/V, bVII, iv) is fine when the measurements show its out-of-key note.
   - Use a 7th (or 6, sus, add9) only when that extra note is clearly in the measurements or is idiomatic in context (a V7 before the tonic, m7 chords in soul/jazz styles). Prefer plain triads otherwise.
   - When the bass is clearly a non-root chord tone, write a slash chord (C/E). Use N.C. where there is no harmony.
   - Repeated sections (verse 2, choruses) normally repeat the same progression; make them consistent unless the measurements clearly differ.
4. MELODY. Write what is sung, in the key.
   - Quantize positions and lengths to sensible note values (16ths at the finest, triplets only if clearly triplet-like). A note's length runs to the next note, unless there's a real gap (breath = rest).
   - Drop or absorb glitches: very short notes at the start/end of a longer note are usually scoops or slides into it.
   - Fix octave jumps that make no melodic sense.
   - Keep accidentals only where the measurement clearly shows a chromatic note.
   - Two words or syllables on the same pitch that the tracker merged into one long note become two notes; use the word timings.
5. LYRICS. One syllable per note, attached with `[lyric=...]`. Split words into syllables with a trailing hyphen: `[lyric=morn-]` then `[lyric=ing]`. A syllable held over several notes (melisma): give the lyric to the first note only and slur the notes (`slur_start` / `slur_end`).
6. SECTIONS. Label the first measure of each section with `"section"`: Intro, Verse 1, Chorus, Bridge, Outro...
7. CLEF. Use "G2" (treble) for most melodies. Use "G2-8" (treble clef sounding an octave lower) for a male voice whose notes sit mostly below C4. In G2-8, still write the real sounding pitches.

# Notation for the melody (same as ScoreScribe's scan format)
""" + _TOKENS + r"""
# Output: a lead sheet JSON object
{"title": "...", "artist": "...", "key": 1, "mode": "major", "time": "4/4", "tempo": 96, "clef": "G2",
 "beat_ql": 1.0, "downbeat_beat": 3,
 "measures": [
   {"section": "Verse 1", "chords": [{"beat": 1, "symbol": "G"}, {"beat": 3, "symbol": "D/F#"}],
    "melody": "B4/4[lyric=Sing] B4/4[lyric=a] D5/4[lyric=sim-] B4/4[lyric=ple]"},
   ...],
 "uncertain": [{"measure": 7, "note": "chord could be D or D7"}]}
- Every measure needs `melody` (use "R" for a bar without singing) and `chords` (an empty list when the chord from the bar before carries on).
- Chord `beat` counts quarter notes from 1 within the bar (2.5 = the "and" of beat 2).
- Melody rhythms must add up to the time signature in every full bar.
- Include every bar from the first sung note (or the first chord) to the end of the song, including instrumental bars (rests + chords).
- List anything you're unsure about in `uncertain`."""

FIRST = ("Here is the analysis of the recording. Make the complete lead sheet following the steps, then submit it "
         "with the submit_leadsheet tool.\n\n")

REVIEW = ("Second pass: harmony and melody review. Below is an automatic check of your lead sheet against the audio "
          "measurements. Go through every ERROR and CHECK item and look at those beats in the analysis again; for "
          "each one, either fix it or keep your version if you're confident the measurement misled. Then review the "
          "harmony as a whole, like a music theorist proofreading an arrangement:\n"
          "- does every chord make sense in the key, and does the progression move logically (tonic, predominant, "
          "dominant, cadences at phrase ends)?\n"
          "- is the harmonic rhythm consistent (chords changing in a steady pattern, not jittering)?\n"
          "- do repeated sections have the same chords and melody where the audio agrees?\n"
          "- does the melody fit the chords: are strong-beat melody notes chord tones, or explainable "
          "suspensions/appoggiaturas?\n"
          "- are the lyrics right (correct obvious mishearings, consistent syllable splitting)?\n"
          "Submit the COMPLETE corrected lead sheet with submit_leadsheet, and summarise what you changed and why in "
          "`changes`.\n\n")

FIX = ("These errors still need fixing before the file can be built. Fix them and submit the COMPLETE lead sheet "
       "again with submit_leadsheet:\n\n")

TOOL_NAME = "submit_leadsheet"


ARRANGE = r"""You are an experienced church musician and arranger. Arrange a song for four-part choir in traditional hymn style (SATB, laid out as in a hymnal: soprano and alto on the treble staff, tenor and bass on the bass staff). The SOPRANO is the song's melody, given in the lead sheet below with its lyrics and chords. Write the ALTO, TENOR and BASS.

# Harmony
- Follow the lead sheet's chords. On each chord the bass sings the chord's bass note: the root, or the slash bass (C/E = E in the bass). Use another inversion only for a smoother bass line, and only when it's idiomatic (passing 6/4, first-inversion IV or ii before V).
- Alto and tenor sing chord tones. Every chord needs its third. In root position, usually double the root; the fifth may be left out. Never double the leading tone.
- The leading tone rises to the tonic (an inner voice may fall to the fifth at a cadence). The seventh of a seventh chord falls by step.
- When the melody has a passing or neighbour note against the chord, the lower voices simply hold the chord.

# Rhythm
- Homophonic, like a hymn: mostly the same rhythm as the soprano, so all four voices sing the syllables together.
- Where the soprano moves in quick notes or a melisma, the lower voices may hold longer notes.
- Rest where the soprano breathes. Every voice must fill every bar exactly.

# Voice leading
- Inner voices move smoothly, mostly by step or common tone. Prefer contrary motion against the bass.
- No parallel fifths or octaves between any two voices (consecutive perfect fifths/octaves in similar motion). No voice crossing.
- Spacing: soprano-alto and alto-tenor within an octave; tenor-bass within a twelfth.
- Ranges (sounding pitch): alto G3-D5 (best A3-C5); tenor C3-A4 (best D3-G4); bass E2-D4 (best G2-C4). The tenor is written at sounding pitch in the bass clef.
- Phrase ends are cadences: make them strong (V-I with the leading tone resolving; plagal IV-I where the song has it).

# Notation
Same token notation as the melody (e.g. `D4/4 C#4/8 B3/8`; rests `r/4`, a silent bar `R`). No lyrics in the lower voices. If the sung melody has ornaments unsuitable for a choir (slides, blue-note bends), you may give a simplified soprano for a bar in `soprano_changes` (measure number -> tokens, keeping the same lyrics); otherwise leave the melody alone.

# Output
{"measures": [{"alto": "...", "tenor": "...", "bass": "..."}, ...],   // exactly one entry per lead-sheet measure, in order
 "soprano_changes": {}}"""

ARRANGE_FIRST = ("Here is the lead sheet (soprano melody with lyrics, and chords). Write the alto, tenor and bass for every "
                 "measure and submit them with the submit_arrangement tool.\n\n")

ARRANGE_REVIEW = ("Part-writing review. Below is an automatic check of your arrangement. Fix every ERROR. For every CHECK "
                  "item (parallel fifths/octaves, missing thirds, notes outside the chord, wrong bass, crossing, spacing, "
                  "range, doubled leading tone), fix it unless it's a deliberate, idiomatic exception. Then read through "
                  "each voice on its own: is it singable and smooth, do the cadences resolve, do repeated sections "
                  "match? Submit the COMPLETE arrangement again with submit_arrangement and summarise your changes in "
                  "`changes`.\n\n")

ARRANGE_TOOL = "submit_arrangement"
