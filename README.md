# config.yaml — CrisperWhisper activated

Two value changes plus one documentation addition, on top of everything
from the previous CrisperWhisper delivery (backend=ct2, compute_type=float32,
etc. — all still in place, unchanged):

```diff
-  backend: mfa
+  backend: crisperwhisper
```
```diff
   crisperwhisper:
-    enabled: false
+    enabled: true
```

Plus a `crisperwhisper —` entry added to the `alignment.backend` option list
at the top of the `alignment:` block, in the same style as the existing
`mfa —`/`whisperx —` entries (it was documented in the
`alignment.crisperwhisper:` sub-block and in `transcribe.py`'s own module
docstring already, but never in *this* list — the first place anyone
setting `backend:` would actually look).

Full diff in `patches/config.yaml.patch`. Same requirement as before: this
needs the image already rebuilt with `crisperwhisper[ct2]` installed (see
the earlier `profanity-hush-crisperwhisper` delivery's `Dockerfile`) — this
config alone doesn't install anything.

## Before running a real job with this

Nothing here has been validated against real audio — same standing caveat
as every delivery in this thread. Recommended first run: Independence Day
again, same as MFA's own validation, checking:
- the 17 known drift cases and the reference SRT, same methodology as before
- `transcript_3.json`'s own text against `transcript_1.json`'s, since this
  is the first config where CrisperWhisper's independently-recognized text
  can actually become `transcript.json` — worth confirming the *words*
  look right, not just the timing, given that's the one respect stage 3
  genuinely differs from stages 1/2
- your real build log for `_check_fork_apis` succeeding cleanly, confirming
  the `ctranslate2` install-order mitigation actually held in your image
