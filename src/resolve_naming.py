#!/usr/bin/env python3
"""
profanity-hush — tiny config query (used by hush.sh)

Not meant to be run by hand. Prints output.naming_style,
output.edition_name, and output.format, one per line as KEY=VALUE, so
hush.sh can:

  - decide whether -- and how -- to redirect a TV episode's output into
    Plex's sibling "Show (Year) {edition-Name}" directory before it ever
    mounts anything for the real per-file run (see hush.sh's
    redirect_for_tv_edition()), and
  - in --batch mode, predict a TV episode's exact target filename (just
    the out_format extension swap -- see steps.mux._output_path()'s
    docstring for why that's *all* it is for TV) to check whether it's
    already there, the same way batch_plan.py already does for movies
    (see that file's module docstring for why TV needs its own,
    bash-side version of this instead).

That decision has to happen in bash, not here: it depends on real,
un-mounted directory names above wherever /input ends up scoped to for
a given file, which nothing running inside a container -- this script
included -- has any visibility into. This script's only job is the one
piece bash genuinely cannot get any other way: what does config.yaml
actually say.
"""
import argparse

import utils
from utils import cfg_get

CONFIG_PATH = "/config/config.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="resolve_naming.py",
        description=(
            "profanity-hush -- prints output.naming_style, "
            "output.edition_name, and output.format as KEY=VALUE lines. "
            "Called by hush.sh; not meant to be run directly."
        ),
    )
    parser.add_argument("--config", default=CONFIG_PATH, metavar="PATH")
    args = parser.parse_args()

    cfg = utils.load_config(args.config)
    utils.validate_config(cfg)

    print(f"naming_style={cfg_get(cfg, 'output', 'naming_style')}")
    print(f"edition_name={cfg_get(cfg, 'output', 'edition_name')}")
    print(f"format={cfg_get(cfg, 'output', 'format')}")


if __name__ == "__main__":
    main()
