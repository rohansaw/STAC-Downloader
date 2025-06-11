import logging
import subprocess

import colorlog
import time


def get_logger():
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",  # %(module)-15s
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "gray",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
        style="%",
    )
    handler = colorlog.StreamHandler()
    handler.setFormatter(formatter)
    logger = colorlog.getLogger("logger")

    if not logger.handlers:
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger

def run_subprocess(cmd: list, step_desc: str, logger):
    """
    Execute a subprocess and stream its output to the console.
    If it fails, log error and raise.
    """
    logger.info(f"Starting: {step_desc}\n  Command: {' '.join(cmd)}")
    t0 = time.time()
    try:
        # Inherit stdout/stderr so user sees real-time output
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during {step_desc}: return code {e.returncode}")
        raise RuntimeError(f"{step_desc} failed (exit code {e.returncode})")
    logger.info(f"Completed: {step_desc} in {time.time() - t0:.2f} seconds")