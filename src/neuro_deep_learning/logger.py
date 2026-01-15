"""App logger."""
import logging
import sys

# Create centralized logger
logger = logging.getLogger("neuro_deep_learning")
logger.setLevel(logging.INFO)

# Avoid adding multiple handlers if they already exist (prevents duplicate logs)
if not logger.handlers:
    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler("app.log")

    # Create formatters
    formatter = logging.Formatter("%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)