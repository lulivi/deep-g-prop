from invoke import task, Collection

from . import docs, tests


#ns = Collection()
#ns.add_collection(docs)
#ns.add_collection(tests)
ns = Collection(docs, tests)

