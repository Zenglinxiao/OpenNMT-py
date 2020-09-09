"""Module for dynamic data transfrom."""
from .transform import make_transforms, get_specials,\
    save_transforms, load_transforms, TransformPipe
from .mics import FilterTooLongTransform, PrefixSrcTransform
from .tokenize import SentencePieceTransform, BPETransform,\
    ONMTTokenizerTransform
from .sampling import SwitchOutTransform, TokenDropTransform,\
    TokenMaskTransform
from .bart import BARTNoiseTransform


AVAILABLE_TRANSFORMS = {
    'sentencepiece': SentencePieceTransform,
    'bpe': BPETransform,
    'onmt_tokenize': ONMTTokenizerTransform,
    'filtertoolong': FilterTooLongTransform,
    'switchout': SwitchOutTransform,
    'tokendrop': TokenDropTransform,
    'tokenmask': TokenMaskTransform,
    'prefix': PrefixSrcTransform,
    'bart': BARTNoiseTransform
}


def get_transforms_cls(transform_names):
    """Return valid transform class indicated in `transform_names`."""
    transforms_cls = {}
    for name in transform_names:
        if name not in AVAILABLE_TRANSFORMS:
            raise ValueError("specified tranform not supported!")
        transforms_cls[name] = AVAILABLE_TRANSFORMS[name]
    return transforms_cls


__all__ = ["get_transforms_cls", "get_specials", "make_transforms",
           "load_transforms", "save_transforms", "TransformPipe"]
