# profanity-hush — CPU-only Docker image
# =========================================================================
# Single build target; no CUDA, no nvidia-container-toolkit needed on host.
#
# Build:
#   docker build -t profanity-hush .
#
# Image size is ~1.1 GB (PyTorch CPU + audio ML stack).
# Model weights (~2-3 GB total) are downloaded on first run and stored in
# the /cache volume — always mount it to avoid re-downloading every run.
# =========================================================================

FROM python:3.11-slim

# Silence debconf "unable to initialize frontend" warnings that appear when
# apt-get runs without a TTY.  Noninteractive is the correct mode for Docker
# builds; this just stops debconf from loudly trying the others first.
ENV DEBIAN_FRONTEND=noninteractive

# ── System packages ────────────────────────────────────────────────────────
# ffmpeg   : audio extraction, muting, audio re-encoding (steps 1, 5, 6, 6b)
# mkvtoolnix : mkvmerge -- final video+audio mux (step 7). Not ffmpeg:
#            real-world testing against full-length Blu-ray rips with many
#            embedded PGS subtitle tracks found ffmpeg's matroska muxer
#            producing files that played back "mostly silent" (or, on
#            Plex, never loading at all) specifically when those subtitle
#            tracks were copied alongside a freshly-supplied audio track
#            from a second input -- ffmpeg's own demuxer already warns
#            "Could not find codec parameters" for several of those same
#            tracks on the way in. mkvmerge handles the identical source
#            tracks (video, all subtitles, chapters) with no such warning
#            and produces a file every player tested (VLC, mpv-based
#            players, Plex) plays correctly -- confirmed by hand against
#            this exact file before switching steps/mux.py over to it.
#            Debian's mkvtoolnix package is CLI-only already (mkvmerge,
#            mkvextract, mkvinfo, mkvpropedit) -- the Qt GUI is the
#            separate mkvtoolnix-gui package, not pulled in here. (Don't
#            confuse this with Ubuntu, which names the CLI-only package
#            mkvtoolnix-cli instead and reserves plain mkvtoolnix for a
#            metapackage that pulls the GUI in too -- this image's base,
#            python:3.11-slim, is Debian, where that split doesn't exist.)
# git      : needed by some pip packages that install from VCS at build time
# libsndfile1 : required by soundfile / librosa (demucs, whisperx deps)
# libgomp1 : OpenMP runtime; demucs benefits from multi-threaded CPU ops
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        mkvtoolnix \
        git \
        libsndfile1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── CPU-only PyTorch ────────────────────────────────────────────────────────
# Install torch/torchaudio BEFORE demucs and whisperx so pip does not pull
# in the much-larger CUDA wheels as a transitive dependency.
# CPU wheels live at a separate index URL; --extra-index-url is needed.
RUN pip install --no-cache-dir \
        torch \
        torchaudio \
        --extra-index-url https://download.pytorch.org/whl/cpu

# ── ML pipeline packages ────────────────────────────────────────────────────
# demucs      : audio source separation (step 2); htdemucs_ft model at runtime
# whisperx    : word-level transcription (step 3); wraps faster-whisper + wav2vec2
# faster-whisper : explicit install to ensure the PyPI version is used, not
#                  a pinned older version pulled by whisperx
# soundfile   : Python wrapper around libsndfile1 (already in the apt layer).
#               torchaudio 2.6+ uses a backend dispatcher for torchaudio.save();
#               without soundfile registered as a backend it raises
#               "Couldn't find appropriate backend to handle uri *.wav".
#               Note: torchaudio 2.9 will switch to torchcodec and won't need
#               soundfile for saving — but having it present won't break anything.
RUN pip install --no-cache-dir \
        demucs \
        whisperx \
        faster-whisper \
        soundfile

# ── Utility packages ────────────────────────────────────────────────────────
# pysrt     : SRT subtitle parsing (step 4, align_srt)
# rapidfuzz : fuzzy string matching for SRT cross-reference
# tqdm      : progress bars for long CPU runs
# pyyaml    : config.yaml parsing
# praatio   : reads MFA's .TextGrid output (steps/align_mfa.py) -- pure
#             Python, pip-installable, no conda/kalpy dependency itself
RUN pip install --no-cache-dir \
        pysrt \
        rapidfuzz \
        tqdm \
        pyyaml \
        praatio

# ── CrisperWhisper (stage 3, comparison-only, see config.yaml's own
#    alignment.crisperwhisper and steps/transcribe_crisperwhisper.py) ───────
# [ct2] extra, not [transformers]: this is crisperwhisper's own preferred
# backend (its backend="auto" resolution tries ct2 first, falling back to
# transformers only if ct2 isn't installed at all -- confirmed by reading
# model.py's _resolve_backend() directly), typically faster on CPU (that's
# CTranslate2's whole purpose -- it's the same engine faster-whisper above
# is built on), and gives full float32 precision with no quality tradeoff
# when compute_type is set explicitly (see config.yaml's own comment on
# alignment.crisperwhisper.compute_type) -- confirmed by reading
# converter.py: float32 is a first-class supported quantization option,
# not a fallback/workaround.
#
# Real risk, confirmed rather than assumed: crisperwhisper[ct2] depends on
# ctranslate2-crisperwhisper (a fork, confirmed against its actual PyPI
# metadata) which installs under the SAME `import ctranslate2` name as the
# plain ctranslate2 faster-whisper already pulled in above (confirmed by
# reading engine.py's own `import ctranslate2` line) -- both write to
# site-packages/ctranslate2/, so whichever installs LAST wins on disk.
# Installed here, after faster-whisper above, specifically so the fork's
# files win. If a future change to this Dockerfile's install order
# accidentally reverses that: this does not fail silently. CrisperWhisper's
# own CT2Engine calls a fail-fast check (_check_fork_apis()) immediately
# after loading the ctranslate2 module and raises a clearly-named error if
# the fork's required methods are missing -- confirmed by reading
# engine.py directly -- so a wrong install order surfaces as a specific,
# diagnosable exception at model-load time, not a silent quality problem.
#
# Model weights (downloaded at runtime, not here -- same lazy-init
# reasoning as MFA's own models below) are under a non-commercial research
# license, NOT MIT the way this inference code is -- see
# steps/transcribe_crisperwhisper.py's own module docstring before
# enabling alignment.crisperwhisper.enabled in config.yaml.
RUN pip install --no-cache-dir "crisperwhisper[ct2]"

# ── Montreal Forced Aligner (default alignment.backend, see config.yaml) ───
# Used in place of whisperx.align() to fix a real, confirmed failure mode:
# whisperx's wav2vec2/CTC aligner faithfully aligns words *within whatever
# segment boundaries WhisperX's own transcribe() already committed to* --
# so if that upstream segment timing is wrong (observed directly: a skipped
# stretch of real dialogue can leave WhisperX several seconds off for
# everything after it), whisperx.align() reproduces the error rather than
# catching it. MFA instead runs a whole-file HMM-GMM search against known
# text, independent of WhisperX's ~30s decode-chunk boundaries entirely --
# see docs/timestamp-drift-investigation.md and steps/align_mfa.py's module
# docstring for the full case this was built against and the validated
# before/after numbers, not just the original finding.
#
# MFA depends on kalpy (Kaldi Python bindings): a compiled extension only
# distributed via conda-forge. Confirmed directly -- `pip install
# montreal-forced-aligner` installs and its pure-Python dependencies
# resolve fine, but importing it fails at runtime with "ModuleNotFoundError:
# No module named '_kalpy'". conda-forge is the only real path.
#
# Installed into its own conda env (not merged into the pip/torch stack
# above) specifically so MFA's own pinned dependency versions can never
# collide with whisperx/demucs/torch's -- these are two isolated Python
# environments on the same image, bridged only by steps/align_mfa.py
# invoking `conda run -n mfa mfa ...` for each call (see that file's
# _mfa_cmd()), never by merging them onto one shared PATH -- an earlier
# version of this tried that and it broke Step 2's own demucs invocation.
#
# This block is required for the default config (alignment.backend: mfa).
# Only skip it (comment this block out and rebuild) if you're setting
# alignment.backend: whisperx everywhere and deliberately accepting the
# drift behavior documented above -- nothing else in this image depends on
# it either way. Adds roughly 300-500MB for the conda env + MFA software
# itself; the larger pretrained-model download (~1-2GB) happens lazily on
# first real use, into /cache, not here -- see the MFA_ROOT_DIR comment
# below for why baking it into the image wouldn't actually help anyway.
RUN apt-get update && apt-get install -y --no-install-recommends \
        bzip2 \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

RUN /opt/conda/bin/conda create -y -n mfa -c conda-forge montreal-forced-aligner \
    && /opt/conda/bin/conda clean -afy

# MFA_ROOT_DIR: where MFA stores its downloaded models, its alignment
# database, and per-run working state.
#
# Pointed at /cache (the same bind-mounted, host-persistent volume
# TORCH_HOME/HF_HOME/NLTK_DATA already use below), NOT baked into the image
# under a path like /opt/mfa_root the way an earlier version of this change
# tried. Two separate problems with that earlier approach, not one:
#
#   1. Model downloads and server/database init were originally RUN here,
#      at build time -- which runs as root. MFA's database backend calls
#      PostgreSQL's initdb, which unconditionally refuses to run as root
#      (confirmed directly: this is exactly the "cannot be run as root"
#      error that approach produced). Building as some other *fixed*
#      non-root user wouldn't actually fix it either -- hush.sh's
#      containers run as an arbitrary, host-determined UID chosen at
#      `docker run` time (`--user "$(id -u):$(id -g)"`, see "Support
#      running as an arbitrary host UID" below), essentially never the
#      same as whatever UID a Dockerfile RUN step used. PostgreSQL data
#      directories are tied to the UID that initialized them, so this
#      needs to happen at actual container-run time, as whatever UID the
#      container really is -- which isn't knowable at build time at all.
#      steps/align_mfa.py's _ensure_mfa_ready() does this lazily, once,
#      the first time a job actually needs MFA, as the correct UID by
#      construction, and is idempotent via its own marker file.
#   2. A path baked into the image doesn't persist across `docker run`
#      invocations the way a bind-mounted volume does -- so even with the
#      UID problem solved, baking downloads in at build time would still
#      mean every single job re-downloads and re-initializes from
#      scratch, since each container gets a fresh copy of the image's own
#      filesystem layer. Pointing at /cache instead means this cost is
#      paid once, ever (whenever the first job that uses alignment.backend:
#      mfa happens to run), exactly matching how whisperx's own models
#      already behave via TORCH_HOME/HF_HOME below -- not a new pattern,
#      the same one.
ENV MFA_ROOT_DIR=/cache/mfa

# Deliberately NOT adding /opt/conda/envs/mfa/bin to PATH here. A conda env's
# bin/ contains a full, separate Python installation -- doing that doesn't
# just make `mfa` discoverable, it shadows `python`/`pip` for every other
# subprocess call anywhere in this image that invokes them by bare name,
# since PATH is searched in order. Confirmed directly: an earlier version of
# this change did exactly that and broke Step 2's own demucs invocation,
# which resolved "python" to the MFA env's interpreter instead of the main
# pip-installed one. steps/align_mfa.py instead calls `conda run -n mfa
# mfa ...` for every invocation, using these two coordinates -- see that
# file's _mfa_cmd() for the full reasoning.
ENV MFA_CONDA_EXE=/opt/conda/bin/conda
ENV MFA_CONDA_ENV=mfa

# ── Redirect all ML cache dirs to /cache (bind-mounted from host) ───────────
# This ensures model weights survive container restarts and aren't
# re-downloaded on each run.  The host path is configured in hush.sh.
#
# Demucs (via torch.hub) → /cache/torch
ENV TORCH_HOME=/cache/torch
# faster-whisper + wav2vec2 alignment models (via huggingface_hub) → /cache/huggingface
ENV HF_HOME=/cache/huggingface
# CrisperWhisper's own model weights are also HuggingFace-hosted (per its
# repo), so this same HF_HOME should already cover them without a separate
# entry here -- expected from how huggingface_hub resolves its cache dir
# globally, not directly confirmed against a real download from this image.
# NLTK punkt tokenizer (used by whisperx internally)
ENV NLTK_DATA=/cache/nltk_data
# Catch-all for any other XDG-respecting cache users
ENV XDG_CACHE_HOME=/cache

# ── Support running as an arbitrary host UID/GID ────────────────────────────
# hush.sh runs the container with `--user "$(id -u):$(id -g)"` so that files
# written into the bind-mounted /jobs, /cache, and /output volumes land on
# the host already owned by the invoking user instead of root.  That UID/GID
# has no /etc/passwd entry inside the image, so four things need handling:
#   1. $HOME must point somewhere writable regardless of UID — anything that
#      isn't already redirected above (matplotlib font cache, stray configs)
#      falls back to $HOME.  World-writable + sticky bit, same pattern as
#      /tmp, so it's safe for any UID without needing a real account.
#   2. Python must not try to write __pycache__/*.pyc next to the read-only,
#      root-owned /app source tree — harmless either way (it silently skips
#      on PermissionError), but disabling it outright is cleaner and avoids
#      depending on that silent-failure behavior.
#   3. Every file under /app must actually be *readable*, and every
#      directory under it *traversable*, by an arbitrary non-root UID/GID —
#      see the chmod after the COPY instructions below.
#   4. PostgreSQL's initdb (alignment.backend: mfa's database server) does
#      its own getpwuid()-style lookup on startup and refuses outright if it
#      can't resolve the current UID to a name — confirmed directly against
#      a real run: "initdb: could not look up effective user ID N: user
#      does not exist". Nothing else in this pipeline needs a real /etc/passwd
#      entry (which is exactly why this wasn't already a solved problem when
#      MFA was added), but this one thing does, unconditionally, with no
#      config flag to turn it off. entrypoint.sh patches one in dynamically
#      at container start, for whatever UID this run actually turns out to
#      be — the only UID that could possibly be right, since it isn't known
#      until then. Only possible because /etc/passwd is made writable by
#      anyone below, same reasoning as /home/hush being 1777.
RUN mkdir -p /home/hush && chmod 1777 /home/hush && chmod 666 /etc/passwd
ENV HOME=/home/hush
ENV PYTHONDONTWRITEBYTECODE=1

# ── Application source ──────────────────────────────────────────────────────
# /app is the root of the Python source tree; steps/ imports utils from here.
ENV PYTHONPATH=/app
WORKDIR /app
COPY src/ /app/

# ── Built-in defaults: config.yaml + word list ──────────────────────────────
# Both of the repo's config/ files are baked into the image so the container
# is fully self-contained: "you can skip installing config files entirely
# and the container uses its built-in defaults" (README install step 3 /
# hush.sh's startup warning) is true for BOTH of them, the same way, via the
# same mechanism -- utils.load_config() falls back to DEFAULT_CONFIG_PATH
# and steps/matching.py falls back to DEFAULT_WORD_LIST_PATH whenever the
# host-mounted /config/config.yaml or /config/word_list.txt isn't present.
#
# This is now the ONLY place a "default" lives for any tunable setting --
# there is no separate, hand-maintained set of Python-side literals to keep
# in sync with config.yaml any more (see utils.cfg_get(), which raises
# rather than silently substituting a hardcoded value if a setting is
# genuinely missing). To change a default: edit config/config.yaml and
# rebuild the image. Nothing in src/*.py needs to change.
COPY config/config.yaml    /app/defaults/config.yaml
COPY config/word_list.txt  /app/defaults/word_list.txt

# `COPY` preserves the exact file-mode bits each source file has in the
# build context — it does NOT guarantee they're world-readable. That
# depends on the contributor's umask, editor, or however the file was last
# saved/transferred on whichever machine `docker build` ran on, and is not
# something this Dockerfile controls. Everything under /app is owned by
# root (no --chown above), so if any file or directory in the build context
# ended up without an "other" read/traverse bit (e.g. mode 600 instead of
# 644), the arbitrary non-root UID from `--user` (point 3 above) gets
# `PermissionError` trying to open it — including, fatally, the entrypoint
# script itself. Force it explicitly rather than relying on every
# contributor's filesystem to happen to produce world-readable files:
#   a+rX  →  read for everyone on files; +traverse (x) only on entries that
#            already have an execute bit somewhere (i.e. directories),
#            so plain .py files don't spuriously become "executable".
RUN chmod -R a+rX /app

COPY entrypoint.sh /app/entrypoint.sh
# 755, not +x: chmod +x is additive -- it only ever adds the execute bit,
# never touches read. If entrypoint.sh's permissions on the host (wherever
# it was saved/created before COPY) didn't already include read access for
# "other", +x alone produces a file that's executable but not *readable* by
# the non-root UID the container actually runs as -- which fails with
# "cannot open ... Permission denied" (confirmed directly against a real
# run), since the shell interpreting the script needs read access to it,
# not just permission to start executing it. An absolute mode doesn't
# depend on whatever the source file's permissions happened to be.
RUN chmod 755 /app/entrypoint.sh

# Declare mount points (documentation only — actual bind mounts are in hush.sh)
VOLUME ["/input", "/output", "/config", "/cache", "/jobs"]

ENTRYPOINT ["/app/entrypoint.sh"]
