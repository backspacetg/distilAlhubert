from .upstream.hubert.hubconf import *
from .upstream.wav2vec2.hubconf import *
from .upstream.alhubert.hubconf import *

def options(only_registered_ckpt: bool = False):
    all_options = []
    for name, value in globals().items():
        torch_hubconf_policy = not name.startswith("_") and callable(value)
        if torch_hubconf_policy and name != "options":
            if only_registered_ckpt and (
                name.endswith("_local")
                or name.endswith("_url")
                or name.endswith("_gdriveid")
                or name.endswith("_custom")
            ):
                continue
            all_options.append(name)

    return all_options
