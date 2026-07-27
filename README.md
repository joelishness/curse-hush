# profanity-hush

Automatically censor profanity from movie files. Feed it a video; get back a censored copy. Designed for unattended overnight CPU-only runs — no GPU required.

---

## How it works

1. **Extract** — Pull the raw audio track from the video (bitstream copy, no re-encode)
2. **Separate** — Demucs (`htdemucs_ft`) splits the audio into a dialog stem and a music/SFX stem
3. **Transcribe** — WhisperX produces word-level timestamps from the dialog stem
4. **Match** — Words are compared against a configurable word list with exact, starts-with, and substring matching
5. **Mute** — Flagged words are silenced only in the dialog stem; music and sound effects play through uninterrupted
6. **Recombine** — The stems are mixed back together and muxed into the output video (video stream is a bit-for-bit copy)

Optionally: cross-reference an SRT subtitle file (Phase 3) or pause for interactive review before muting (available now via `--interactive`).

---

## Requirements

- **Docker** — tested on Linux (Manjaro/Arch and Ubuntu). Docker Desktop on macOS and Windows should work but is untested.
- **~4-5 GB of free disk** for model weights (downloaded on first run, cached afterward) — includes MFA's pretrained models now that it's the default alignment backend
- **16 GB RAM** recommended — Demucs is memory-hungry. See [Expected Runtimes](#expected-runtimes).
- No GPU needed.

---

## Installation

### 1. Build the Docker image

```bash
git clone https://github.com/yourname/profanity-hush.git
cd profanity-hush
docker build -t profanity-hush .
```

Image size is approximately 1.5 GB (includes the conda environment for MFA, the default alignment backend). Model weights (~4-5 GB total, including MFA's own) are downloaded on the first run and cached — always use a persistent cache directory (see below).

Using `docker compose`:

```bash
docker compose build
```

### 2. Make `hush.sh` executable

```bash
chmod +x hush.sh
```

### 3. (Optional) Install config files

The pipeline ships with working defaults. To customize, copy the config to your user config directory:

```bash
mkdir -p ~/.config/profanity-hush
cp config/config.yaml ~/.config/profanity-hush/
cp config/word_list.txt ~/.config/profanity-hush/
```

If you skip this step, `hush.sh` will warn that the config directory is empty and the container will use its built-in defaults. For a real run you'll want your own `word_list.txt`.

---

## Usage

```
hush.sh [OPTIONS] <input_video> [subtitle_file]
hush.sh --batch [--recursive] [OPTIONS] <input_dir>

Options:
  -o, --output DIR      Output directory (default: same directory as input)
  -c, --config DIR      Config directory (default: ~/.config/profanity-hush)
      --cache  DIR      Model cache directory (default: ~/.cache/profanity-hush)
      --jobs   DIR      Job history directory (default: ~/.local/share/profanity-hush/jobs)
      --interactive     Pause for review of flagged words before muting
      --no-interactive  Force unattended mode (overrides config.yaml)
      --keep-tmp        Keep large intermediate WAV stems after the run
  -b, --batch           Process every video file in <input_dir> in sequence —
                        see "Batch Processing" below
  -r, --recursive       With --batch, also descend into subdirectories
      --skip-index N    Correct a false positive — see "Correcting Mistakes" below
      --add-interval TEXT START END
                        Correct a false negative — START/END as seconds or
                        H:MM:SS.mmm — see "Correcting Mistakes" below
      --redo-review     Re-enter interactive review on an already-completed job
      --redo-step STEP  Force a single step to redo on an existing job — see
                        "Re-running a Single Step" below
      --dry-run         Print the docker command without executing it
  -h, --help
```

### Examples

```bash
# Basic — censor a film, output alongside the input
./hush.sh movie.mkv

# With an SRT file for improved accuracy (Phase 3, coming soon)
./hush.sh movie.mkv movie.srt

# Review flagged words before committing to a muted output
./hush.sh --interactive movie.mkv

# Send output to a specific directory
./hush.sh -o ~/censored/ movie.mkv

# Preview what docker command would run (no processing)
./hush.sh --dry-run movie.mkv
```

By default, the output file is named for Plex's `{edition-Name}` convention. Movies get it right after the release year, so Plex shows it as a selectable Edition of the same movie:
`Movie (1986).sd.hevc.mkv` → `Movie (1986) {edition-Hushed}.sd.hevc.mkv`
TV episodes get it at the end of the episode title instead, since a movie-style "right after the year" would land it in the middle of the filename (the year there belongs to the series, not the episode):
`Psych (2006) - s02e01 - American Duos.sd.hevc.mkv` → `Psych (2006) - s02e01 - American Duos {edition-Hushed}.sd.hevc.mkv`
Set `output.naming_style: suffix` in `config.yaml` for a plain suffix instead: `movie.mkv` → `movie_censored.mkv`.

### Batch Processing

`--batch` processes every video file in a directory, one after another, instead of one `hush.sh` invocation per file:

```bash
# One season — top-level files only, no subdirectories
./hush.sh --batch "Psych (2006)/Season 02"

# A whole show — every season + Specials, in one command
./hush.sh --batch --recursive "Psych (2006)"

# Redirect the whole batch's output elsewhere, mirroring subdirectories under it
./hush.sh --batch --recursive -o ~/censored/ "Psych (2006)"
```

A file already having a censored output next to it (e.g. a `{edition-Hushed}` sibling) is skipped automatically — judged by the output file itself, not local job history, since a large library is often built up across more than one machine. Re-running the same `--batch` command later — after adding new episodes, after an interrupted run, after fixing a failure — only processes what's still missing; nothing gets redone. `Ctrl-C` stops the batch after the file currently in progress finishes its current step (that file resumes from there next time, same as any interrupted single-file run — see "Job History" below); an ordinary per-file failure is logged and the batch continues on to the next file, with a summary of anything that failed at the end.

`--batch` can't be combined with `--skip-index` / `--add-interval` / `--redo-review` / `--redo-step` — those target one already-completed job, not a directory; run them against that one file directly instead. `--interactive` does work with `--batch`, but pauses for review on every file in the queue, one after another — usually only worth combining for a small batch.

Every `--batch` run also writes a high-level overview to `<jobs_dir>/batch-logs/`, one file per invocation: the plan phase's results (how many files were found, already done, and queued), and each file's start time, end time, and duration. It deliberately does *not* duplicate each file's own full step-by-step transcript — that already lives in that job's own `job_dir/logs/*.log` — so it stays a quick, scannable summary across 100+ files rather than growing as long as reading through every job individually. It does still flag anything notable in one short line, e.g. an MFA alignment falling back to `whisperx.align()` for a segment, using a compact result line pipeline.py prints for exactly this purpose. Set `AC_LOG_LEVEL=debug` to have the plan phase name the specific files it skipped or queued, not just the counts.

### Using docker compose (workstation)

```bash
# Copy and edit the environment file
cp .env.example .env
# Edit .env — at minimum set INPUT_DIR to the directory containing your video.
# Also set HOST_UID/HOST_GID (see comments in .env.example) — unlike hush.sh,
# docker compose can't detect these automatically, and skipping them leaves
# job/cache/output files owned by root on the host.

# Run
docker compose run --rm hush /input/movie.mkv
docker compose run --rm hush /input/movie.mkv --interactive
```

---

## Expected Runtimes

CPU-only processing is intentionally slow — runs are queued overnight.

Measured on a 16 GB machine, `htdemucs_ft`, `large-v2` WhisperX:

| Stage | Per segment (30 min) | For a 2-hour film |
|---|---|---|
| Demucs `--shifts 1` (default) | ~33 min | ~2.2 hours (4 segments) |
| Demucs `--shifts 4` (quality) | ~2.2 hours (estimated†) | ~8.7 hours (estimated†) |
| WhisperX `large-v2` | — | ~30–60 min total |
| **Total (shifts=1)** | | **~3–4 hours** |

† `shifts=4` runtime is a linear extrapolation from the measured `shifts=1` figure;
  it has not yet been directly timed.

`shifts=1` is the default. It produces good quality output. `shifts=4` may improve
quality further at roughly 4× the compute cost — use it only if you have time to
spare and have validated the quality difference is meaningful on your content
(see `config.yaml`).

**Memory:** Peak memory is bounded per segment by the 30-minute segment size (default).
Reduce `audio.segment_size_sec` in `config.yaml` if you see OOM errors on machines
with less than 16 GB RAM. Still hitting this on a bigger machine, or inconsistently?
See [Troubleshooting](#troubleshooting) — RAM size isn't the whole story.

---

## Configuration

### `config/config.yaml`

The main settings file. Key options:

```yaml
demucs:
  model: htdemucs_ft   # htdemucs | htdemucs_ft | htdemucs_6s
  shifts: 1            # 1 = fast/default, 4 = quality, 10 = maximum

whisperx:
  model: large-v2      # large-v2 (recommended) | medium | small
  language: en         # ISO 639-1; null for auto-detect

alignment:
  backend: mfa         # mfa (default, validated) | whisperx (fallback / opt-out).
                       # Fixes word *timing* for words already recognized; does not
                       # change what gets recognized -- see docs/timestamp-drift-investigation.md
                       # for the validated before/after numbers and what's still open.

audio:
  segment_size_sec: 1800   # 30 min per segment; reduce if OOM

censoring:
  method: mute         # mute | beep (beep is not yet implemented -- selecting
                       # it fails the run at Step 5 with a clear error)
  padding_ms: 50       # silence added before/after each word (ms)

output:
  naming_style: plex_edition   # plex_edition | suffix -- see Usage examples above
  format: mkv                  # mkv | mp4
  keep_intermediates: false
  keep_correction_artifacts: true   # keeps dialog.wav/score_sfx.wav so corrections stay cheap
```

See the full file at `config/config.yaml` for all options and their documentation.

### Environment variable overrides

| Variable | Effect |
|---|---|
| `AC_LOG_LEVEL` | `debug` / `info` / `warning` |
| `AC_KEEP_INTERMEDIATES=1` | Keep all large WAV stems after run (same as `--keep-tmp`) |
| `AC_KEEP_CORRECTION_ARTIFACTS=0` | Don't keep `dialog.wav`/`score_sfx.wav` either (these default to kept — see [Correcting Mistakes](#correcting-mistakes)) |
| `AC_INTERACTIVE=1` | Enable interactive review (same as `--interactive`) |
| `AC_SEGMENT_SIZE` | Override `audio.segment_size_sec` in seconds; `0` disables segmentation |
| `AC_TZ_OFFSET` / `AC_TZ_NAME` | Host UTC offset (e.g. `-0700`) / cosmetic abbreviation (e.g. `PDT`) used for log timestamps. `hush.sh` sets both automatically from the host's clock — see [Logging](#logging) below — only needed by hand if you're running the container some other way. |
| `AC_INPUT_HOST_DIR` / `AC_OUTPUT_HOST_DIR` | Real host directories for the input/output files, so `job.json` is readable outside the container. `hush.sh`/`docker compose` set both automatically — see [Logging](#logging) below — only needed by hand if you're running the container some other way. |

```bash
AC_LOG_LEVEL=debug ./hush.sh movie.mkv
AC_SEGMENT_SIZE=900 ./hush.sh movie.mkv   # 15-min segments
```

### `config/word_list.txt`

One entry per line. Comments (`#`) and blank lines are ignored. Supports exact, starts-with, substring, and case-sensitive matching:

| Notation | Matches |
|---|---|
| `word` | Exact, case-insensitive |
| `=word` | Exact, case-sensitive — useful for distinguishing profanity from proper nouns |
| `word*` | Starts-with, case-insensitive |
| `*word*` | Substring, case-insensitive |

WhisperX capitalizes proper nouns naturally, so `=dick` catches the profane usage while leaving `Dick` (a name) untouched.

---

## Logging

Console timestamps automatically match this machine's local clock: `hush.sh` detects your current UTC offset (`date +%z`/`%Z`) and forwards it into the container, so every log line is stamped in your own time zone instead of the container's default UTC. If that detection ever fails (or you're running the container some other way — see `AC_TZ_OFFSET`/`AC_TZ_NAME` above), timestamps fall back to UTC, and are clearly labelled as such rather than looking like local time that's quietly several hours off.

```
2026-06-25 16:25:41 -0700 [INFO ] [separate ] [2/4] dialog.wav (412 MB) ...
```

`job.json`'s `started_at`/`failed_at`/`completed_at` fields stay in UTC (ISO 8601, with a `+00:00` offset) — useful for comparing job records regardless of which time zone a given run happened to log in — alongside `*_local` companions for convenience when reading the file directly. The job folder's own leading timestamp (see [Job History](#job-history) below) uses the same local time as everything else above, for the same reason: it's a place you're likely to actually look (browsing the jobs folder directly), so it should read as what your clock said, not require doing offset arithmetic.

`job.json`'s `input_path` and `mux.output_path` are similarly host-aware: `hush.sh`/`docker compose` also forward the real directories they're already using for the `-v` mounts (`AC_INPUT_HOST_DIR`/`AC_OUTPUT_HOST_DIR` — see [Configuration](#configuration) above), so those fields show somewhere you can actually navigate to (e.g. `/nas/media/movies/Movie (1986)/`) instead of the container's own `/input`/`/output` mount points, which mean nothing outside it.

---

## Job History

Every run creates a job record under `~/.local/share/profanity-hush/jobs/`, in a folder named `YYYYMMDD_HHMMSS_<movie-slug>_<hex8>` (the timestamp is your local time — see [Logging](#logging) above — and the slug makes it easy to spot the right job by filename without opening anything). The merged `transcript.json` and the censor log are always preserved, along with `dialog.wav` and `score_sfx.wav` (the pre-mute audio stems) — together these are what makes [correcting a mistake](#correcting-mistakes) after watching the film fast, without repeating the expensive separation and transcription steps. (The per-segment `transcript_NN.json` files WhisperX writes on the way to `transcript.json` are cleaned up once merged, same as the other per-segment intermediates — pass `--keep-tmp` if you want to inspect them.)

Large intermediate WAV files are deleted by default once each is no longer needed. Pass `--keep-tmp` to retain all of them (including ones not needed for corrections); see `output.keep_correction_artifacts` in `config.yaml` to control just the two needed for corrections independently.

Files are written owned by the user who ran `hush.sh`, not root. If you have job or cache files from before this was fixed, they'll still be owned by root — clean them up once with:

```bash
sudo chown -R "$(id -u):$(id -g)" \
    ~/.local/share/profanity-hush \
    ~/.cache/profanity-hush
```

If you also ran with a custom `-o`/`--output` directory before this was fixed, chown that too — its location isn't fixed the way the job/cache paths above are (it defaults to the same directory as your input video, or wherever `-o` pointed), so it isn't included in the command above:

```bash
sudo chown -R "$(id -u):$(id -g)" /path/to/your/output/dir
```

---

## Interactive Review

Pass `--interactive` to pause before muting and review each flagged word:

```
[3 of 11]  Word: "crap"  |  Confidence: 0.94  |  Time: 0:23:14.800 – 0:23:15.100
Context: "...and then he said crap right in front of..."
Action? [Y]es / [N]o / [A]dd word / [S]kip rest / [Q]uit  >
```

- **Y** — approve; word will be muted (default)
- **N** — reject; word will not be muted
- **A** — add a missed word/phrase: searches the transcript for it first (picks automatically if there's one match, lets you choose if there are several); falls back to manual timestamp entry if it's not found at all — accepts either raw seconds (`1203.14`) or `H:MM:SS.mmm` (`0:20:03.140`)
- **S** — approve all remaining without prompting
- **Q** — abort without writing output

Requires a real terminal. `hush.sh --interactive` allocates one automatically; running the container directly needs `-it` on `docker run`.

---

## Correcting Mistakes

This is the expected day-to-day workflow: run unattended, watch the film (maybe with the people it was censored for), and fix anything wrong afterward — without waiting through separation and transcription again.

**False positive** (a word got muted that shouldn't have been — e.g. WhisperX mis-hearing dialogue): find the entry in that job's `censor_log.json` (under `~/.local/share/profanity-hush/jobs/<job-folder>/` — see [Job History](#job-history) for the folder naming) by its timestamp — `start_hms`/`end_hms` are in `H:MM:SS.mmm`, the same notation your media player's seek bar/goto-time field uses, so you can jump straight to the moment instead of doing the seconds-to-minutes math by hand — and note its `word_index`:

```json
{
  "source": "matched",
  "word": "hell",
  "entry": "hell",
  "word_index": 4856,
  "start": 5275.01,
  "start_hms": "1:27:55.010",
  "end": 5275.23,
  "end_hms": "1:27:55.230",
  "padded_start": 5274.96,
  "padded_start_hms": "1:27:54.960",
  "padded_end": 5275.28,
  "padded_end_hms": "1:27:55.280",
  "score": 0.97
}
```

(`padded_start`/`padded_end` above reflect this job's `padding_ms: 50` default — `start`/`end` widened by 0.05s on each side, which is what's actually muted; `score` is WhisperX's word-level confidence for the transcribed word, shown here as a representative value.)

(In this real example, WhisperX had transcribed the line "Ned! Land!" as "What the hell" — a transcription error, not a word-list problem; the word list correctly matched the literal text WhisperX produced.) Then:

```bash
./hush.sh --skip-index 4856 movie.mkv
```

**False negative** (something that should have been muted wasn't): note the timestamp while watching, then pass it to `--add-interval` as either raw seconds or `H:MM:SS.mmm` (whichever's easier to read off your player):

```bash
./hush.sh --add-interval "missed word" 1203.14 1203.48 movie.mkv
./hush.sh --add-interval "missed word" 0:20:03.140 0:20:03.480 movie.mkv   # same interval, H:MM:SS.mmm
```

Both flags are repeatable and combinable in one run (and the two timestamp notations can be mixed freely, even within the same `--add-interval`):

```bash
./hush.sh --skip-index 4856 --skip-index 412 --add-interval "oops" 88.0 88.4 movie.mkv
```

Re-running on the same input file (same path, unchanged) automatically finds the existing job — no job ID to look up or pass. Only the muting, recombining, and muxing steps (5–7) redo; typically a couple of minutes, not the original multi-hour run. This depends on `dialog.wav`/`score_sfx.wav` still being in the job directory, which is the default (`output.keep_correction_artifacts: true`) — if you've set that to `false`, or `--keep-tmp` wasn't used before that setting existed, the fix needs a full re-run instead, and `hush.sh` will say so clearly rather than silently doing the expensive thing.

**Prefer a fuller second look instead?** `--redo-review` re-enters the interactive review loop from scratch (every flagged word, not just the one you noticed) on an already-completed job:

```bash
./hush.sh --redo-review movie.mkv
```

This can't be combined with `--skip-index`/`--add-interval` in the same run — it rewrites the review file from scratch and would discard direct edits made moments earlier.

---

## Re-running a Single Step

`--skip-index`/`--add-interval`/`--redo-review` above are for fixing a *content* mistake — the word list or transcript got something wrong. `--redo-step` is for a different situation: you've changed how a **step itself** works (swapped the muxer, tuned a mute padding value, fixed an encode setting) and want to re-test that change against a job that already finished, without re-running everything before it:

```bash
./hush.sh --redo-step 7_mux movie.mkv
```

Repeatable, and valid for `4b_flag`, `4b_review`, `5_mute`, `6_recombine`, `6b_encode`, and `7_mux`. Steps 1a–3b aren't offered: they're resumed as a single atomic block, and their per-segment intermediates may already be gone, so redoing one of them alone isn't safe. `--redo-step` never touches `review.json` and can't be combined with `--skip-index`/`--add-interval`/`--redo-review` in the same run.

It also requires the job to actually be found first: if `compute_job_id()` doesn't land on an existing job for this input file (same path, unchanged), `hush.sh` refuses with a clear error rather than silently falling through to a full from-scratch run. This is also why hand-editing `steps_completed` in `job.json` directly isn't recommended, even though each step does check its own entry independently and the edit can appear to work: a single stray character (a trailing comma is the classic one) makes the whole file invalid JSON, and an unparseable `job.json` looks identical to "no job exists yet" to the code that's trying to find it — the visible symptom is a full multi-hour re-run with no explanation, not an error. `--redo-step` is the safe, validated way to get the same result.

---

## Troubleshooting

### Step 2 (Demucs) fails with "Command failed (exit -9)"

```
[ERROR] [separate ] Step 2 failed: Command failed (exit -9)
```

with no traceback — the process just stops. `exit -9` is SIGKILL: the OOM killer (or a container memory limit) killing `demucs`, not a code or input-file problem.

RAM size alone doesn't fully protect against this — it depends on what else is using memory at the time, so the crash point can shift between runs, or disappear after a plain **reboot** with nothing else changed. Confirm with `docker inspect <container> --format='{{.State.OOMKilled}}'` or `dmesg -T | grep -i "killed process"`.

**To fix:** reboot or free other memory first; otherwise lower `audio.segment_size_sec` (or `AC_SEGMENT_SIZE`) below the 1800s default (see [Configuration](#configuration)), switch `demucs.model` to `htdemucs` instead of `htdemucs_ft` (~1/4 the memory, at some quality cost), or lower `demucs.shifts`.

---

## Limitations

- **Stereo output only (v1):** Multi-channel surround audio (5.1, 7.1) is downmixed to stereo for processing. The original audio is preserved in the job store; per-channel processing is planned for a future version.
- **Separation artifacts:** Demucs is excellent but not perfect — some bleed between stems is expected, especially in dense action scenes.
- **Context-blind matching:** The word list has no understanding of usage context. `=dick` / `Dick` case distinction is the primary mitigation; interactive review handles the rest.
- **v1 processes only the primary audio track.** Commentary tracks and alternate language tracks in the source container are dropped.
- **Densely repetitive dialogue can still slip through, regardless of alignment backend.** A line repeated verbatim multiple times in a row (confirmed on one real film) can cause WhisperX's own recognition to drop it entirely — no alignment backend can place a word that was never recognized as text in the first place. See `docs/timestamp-drift-investigation.md` for the one specific case this was traced down to, and why it's now understood to be a recognition gap rather than a timing one.

---

## License

[GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0). In particular, this means that if you run a modified version of this project as a network service that other users interact with, you must make the source of your modified version available to them under the same license — see the full license text for the precise terms.
