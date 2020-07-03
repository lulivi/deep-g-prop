"""Common globals for the code."""
from os import environ

SEED = int(environ.get("SEED", 12345))
