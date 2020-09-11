from onmt.utils.logging import logger
from onmt.dynamic.transforms import register_transform
from .transform import Transform


@register_transform(name='filtertoolong')
class FilterTooLongTransform(Transform):
    """Filter out sentence that are too long."""

    def __init__(self, opts):
        super().__init__(opts)
        self.src_seq_length = opts.src_seq_length
        self.tgt_seq_length = opts.tgt_seq_length

    @classmethod
    def add_options(cls, parser):
        """Avalilable options relate to this Transform."""
        group = parser.add_argument_group('Transform/Filter')
        group.add('--src_seq_length', '-src_seq_length', type=int, default=200,
                  help="Maximum source sequence length")
        group.add('--tgt_seq_length', '-tgt_seq_length', type=int, default=200,
                  help="Maximum target sequence length")

    def apply(self, src, tgt, stats=None, **kwargs):
        """Return None if too long else return as is."""
        if len(src) > self.src_seq_length or len(tgt) > self.tgt_seq_length:
            if stats is not None:
                stats.filter_too_long()
            return None
        else:
            return src, tgt

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}, {}={}'.format(
            'src_seq_length', self.src_seq_length,
            'tgt_seq_length', self.tgt_seq_length
        )


@register_transform(name='prefix')
class PrefixSrcTransform(Transform):
    """Add target language Prefix to src sentence."""

    def __init__(self, opts):
        super().__init__(opts)
        self.prefix_dict = self.get_prefix_dict(self.opts)

    @staticmethod
    def _get_prefix(corpus):
        """Get prefix string of a `corpus`."""
        prefix_tag_format = "｟_tgt_is_{}｠"
        if 'prefix' in corpus['transforms']:
            tl = corpus['tgt_lang']
            prefix = prefix_tag_format.format(tl)
        else:
            prefix = None
        return prefix

    @classmethod
    def get_prefix_dict(cls, opts):
        """Get all needed prefix correspond to corpus in `opts`."""
        prefix_dict = {}
        for c_name, corpus in opts.data.items():
            prefix = cls._get_prefix(corpus)
            if prefix is not None:
                logger.info(f"Get prefix for {c_name}: {prefix}")
                prefix_dict[c_name] = prefix
        return prefix_dict

    @classmethod
    def get_specials(cls, opts):
        """Get special vocabs added by prefix transform."""
        prefix_dict = cls.get_prefix_dict(opts)
        src_specials = set(prefix_dict.values())
        return (src_specials, set())

    def _prepend(self, tokens, corpus_name):
        """Get prefix according to `corpus_name` and prepend to `tokens`."""
        corpus_prefix = self.prefix_dict.get(corpus_name, None)
        if corpus_prefix is None:
            raise ValueError('corpus_name does not exist.')
        return [corpus_prefix] + tokens

    def apply(self, src, tgt, stats=None, **kwargs):
        """Prepend prefix to src side tokens."""
        corpus_name = kwargs.get('corpus_name', None)
        if corpus_name is None:
            raise ValueError('corpus_name is required.')
        src = self._prepend(src, corpus_name)
        return src, tgt

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}'.format('prefix_dict', self.prefix_dict)
