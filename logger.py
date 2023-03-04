import logging


def get_logger(name, level=logging.INFO):
    logr = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(levelname)s:%(asctime)s: %(message)s", datefmt="%H:%M:%S")
    )
    logr.addHandler(handler)
    logr.setLevel(level)
    logr.propagate = False  # Prevent the modal client from double-logging.
    return logr


log = get_logger(__name__)
