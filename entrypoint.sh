#!/bin/sh
# See Dockerfile's "Support running as an arbitrary host UID/GID" section
# (item 4) for the full explanation. Short version: hush.sh runs this
# container with --user "$(id -u):$(id -g)", an arbitrary UID with no
# /etc/passwd entry -- fine for everything in this pipeline except
# PostgreSQL's initdb (alignment.backend: mfa's database server), which
# does its own getpwuid()-style lookup and refuses outright if it can't
# resolve the current UID to a name. Confirmed directly against a real
# run, not theoretical: "initdb: could not look up effective user ID N:
# user does not exist".
#
# Give it one, dynamically, matching this container's actual runtime UID --
# the only UID that could possibly be correct, since it isn't known until
# right now. /etc/passwd is chmod 666 at build time specifically so this
# can happen without root.
#
# Safe to run every time: only touches things if this exact UID doesn't
# already have an entry, so it's a no-op after the first call within a
# container's lifetime, and never conflicts with a UID that's already
# resolvable (e.g. root, uid 0, if this ever runs that way).
set -e

if ! getent passwd "$(id -u)" > /dev/null 2>&1; then
    echo "hush:x:$(id -u):$(id -g)::${HOME:-/home/hush}:/bin/sh" >> /etc/passwd
fi

exec python /app/pipeline.py "$@"
