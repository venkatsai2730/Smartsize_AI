from loguru import logger
import sys

def setup_logging():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time} {level} {message}")
    logger.add("logs/app.log", rotation="1 MB", level="DEBUG")