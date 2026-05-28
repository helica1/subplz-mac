import sys

from .logger import configure_logging, logger
from .config import load_config
from .helpers import find, rename, copy, extract
from .batch import run_batch
from .sync import run_sync
from .gen import run_gen
from .srs import run_srs

# --- CHANGED: We now import get_args and create_structured_inputs separately ---
from .cli import get_args, get_inputs
from .auto.watcher import run_watcher
from .auto.scanner import run_scanner


def _run_tts(args):
    # Lazy import: tts.py pulls in numpy/torch/SBV2, which we don't want
    # to load for unrelated subcommands like `find` or `rename`.
    from .tts import run_tts
    return run_tts(args)


def _run_voice(args):
    from .tts import run_voice
    return run_voice(args)


def execute_on_inputs():
    """
    Parses CLI arguments, loads config, combines them, configures logging,
    and dispatches to the correct handler function.
    """

    args = get_args()
    # Some subcommands (tts, voice) don't expose --config; load defaults in that case.
    config = load_config(getattr(args, "config", None))
    inputs = get_inputs(args, config)
    inputs.config_data = config
    configure_logging(config)

    COMMAND_MAP = {
        "watch": run_watcher,
        "scanner": run_scanner,
        "find": lambda args: find(args.dirs),
        "rename": rename,
        "copy": copy,
        "extract": extract,
        "batch": run_batch,
        "sync": run_sync,
        "gen": run_gen,
        "srs": run_srs,
        "tts": _run_tts,
        "voice": _run_voice,
    }

    handler_function = COMMAND_MAP.get(inputs.subcommand)

    if handler_function:
        try:
            handler_function(inputs)
        except Exception as e:
            # This correctly logs the error with the full traceback
            logger.opt(exception=True).critical(
                f"A fatal error occurred during the '{inputs.subcommand}' command. {e}"
            )
            sys.exit(1)
    else:
        logger.error(f"Error: Unknown subcommand '{inputs.subcommand}'")
        sys.exit(1)
