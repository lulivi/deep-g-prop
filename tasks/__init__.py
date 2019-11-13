from invoke import Collection

from . import docs, tests

ns = Collection(docs, tests)
