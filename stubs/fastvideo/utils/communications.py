"""Stand-in for FastVideo sequence-parallel helpers.

AniSora imports these names while loading WanModel. Single-GPU inference
does not call them.
"""


def _unused(*_args, **_kwargs):
    raise RuntimeError("sequence parallel is not enabled")


gather_sequence = _unused
split_sequence = _unused
all_to_all_4D = _unused
