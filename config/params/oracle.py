import inspect
from rlutils.observers.pbrl.interfaces import OracleInterface
from .. import oracles

P = {
    "pbrl": {
        "interface": {
            "class": OracleInterface,
            # https://stackoverflow.com/a/63449487
            "oracle": {name: func for name, func in oracles.__dict__.items()
                       if inspect.isfunction(func) and inspect.getmodule(func) == oracles}
        }
    }
}
