# -*- coding: utf-8 -*-
from functools import partial

import six
import torch
from torchtext.data import Field, RawField

from onmt.inputters.datareader_base import DataReaderBase


class TextDataReader(DataReaderBase):
    def __init__(self, n_src_ctxs, n_tgt_ctxs):
        self.n_src_ctxs = n_src_ctxs
        self.n_tgt_ctxs = n_tgt_ctxs

    @classmethod
    def from_opt(cls, opt):
        return cls(n_src_ctxs=opt.n_src_ctxs, n_tgt_ctxs=opt.n_tgt_ctxs)

    def read(self, sequences, side, _dir=None):
        """Read text data from disk.

        Args:
            sequences (str or Iterable[str]):
                path to text file or iterable of the actual text data.
            side (str): Prefix used in return dict. Usually
                ``"src"`` or ``"tgt"``.
            _dir (NoneType): Leave as ``None``. This parameter exists to
                conform with the :func:`DataReaderBase.read()` signature.

        Yields:
            dictionaries whose keys are the names of fields and whose
            values are more or less the result of tokenizing with those
            fields.
        """
        assert _dir is None or _dir == "", \
            "Cannot use _dir with TextDataReader."
        if isinstance(sequences, str):
            sequences = DataReaderBase._read_file(sequences)
        if side == 'src' and self.n_src_ctxs != 0:
            return self.read_with_ctx(sequences, 'src', self.n_src_ctxs)
        elif side == 'tgt' and self.n_tgt_ctxs != 0:
            return self.read_with_ctx(sequences, 'tgt', self.n_tgt_ctxs)
        else:
            return self.read_naive(sequences, side)

    def read_naive(self, sequences, side):
        """Yield each line with its index."""
        for i, seq in enumerate(sequences):
            if isinstance(seq, six.binary_type):
                seq = seq.decode("utf-8")
            yield {side: seq, "indices": i}

    def read_with_ctx(self, sequences, side, n_ctx):
        """Yield each segment with `n_ctx` previous contexts.

        This will read from a document corpus where a blankline as boundary.
        boudary line will be skiped.
        """
        seqs, idxs = [], []
        for i, seq in enumerate(sequences):
            if isinstance(seq, six.binary_type):
                seq = seq.decode("utf-8")
            if seq.rstrip('\n') == '':
                if len(seqs) != 0:
                    sentexs = docex2sentexs((seqs, idxs), n_ctxs=n_ctx)
                    for sent, ctxs, idx in sentexs:
                        yield {side: (sent, ctxs), "indices": idx}
                seqs, idxs = [], []
            else:
                seqs.append(seq)
                idxs.append(i)
        if len(seqs) != 0:
            sentexs = docex2sentexs((seqs, idxs))
            for sent, ctxs, idx in sentexs:
                # yield {side: sent, side+"_ctxs": ctxs, "indices": idx}
                yield {side: (sent, ctxs), "indices": idx}


def docex2sentexs(docex, n_ctxs=2, pad=''):
    """Convert Doc-like example to sentence examples with contexts."""
    seqs, idxs = docex
    sentex = []
    pad_seqs = [pad] * n_ctxs + seqs
    for i, idx in enumerate(idxs):
        ctxs = pad_seqs[i:i+n_ctxs]
        seq = pad_seqs[i+n_ctxs]
        sentex.append((seq, ctxs, idx))
    return sentex


def text_fields_len(ex, side, bos=False, eos=False):
    """Count the len(`ex`.`side`), optional return size with `bos`, `eos`."""
    def _text_field_len(text_field, bos=False, eos=False):
        n_tokens = len(text_field[0])
        if bos:
            n_tokens += 1
        if eos:
            n_tokens += 1
        return n_tokens

    assert side in ['src', 'tgt'], "only src or tgt text_field is countable."
    text_field = getattr(ex, side)
    if isinstance(text_field, tuple):
        n_tokens_multi = ([_text_field_len(text_field[0], bos, eos)] +
                          [_text_field_len(ctx_ff, bos, eos)
                          for ctx_ff in text_field[1]])
    else:
        n_tokens_multi = [_text_field_len(text_field, bos, eos)]
    return n_tokens_multi


def text_sort_key(ex):
    """Sort using the number of tokens in the sequence."""
    src_len = text_fields_len(ex, "src")
    if hasattr(ex, "tgt"):
        tgt_len = text_fields_len(ex, "tgt")
        src_base_len, src_ctxs_lens = src_len[0], src_len[1:]
        tgt_base_len, tgt_ctxs_lens = tgt_len[0], tgt_len[1:]
        all_lens = [src_base_len, tgt_base_len] + src_ctxs_lens + tgt_ctxs_lens
        return all_lens
    return src_len


# mix this with partial
def _feature_tokenize(
        string, layer=0, tok_delim=None, feat_delim=None, truncate=None):
    """Split apart word features (like POS/NER tags) from the tokens.

    Args:
        string (str): A string with ``tok_delim`` joining tokens and
            features joined by ``feat_delim``. For example,
            ``"hello|NOUN|'' Earth|NOUN|PLANET"``.
        layer (int): Which feature to extract. (Not used if there are no
            features, indicated by ``feat_delim is None``). In the
            example above, layer 2 is ``'' PLANET``.
        truncate (int or NoneType): Restrict sequences to this length of
            tokens.

    Returns:
        List[str] of tokens.
    """

    tokens = string.split(tok_delim)
    if truncate is not None:
        tokens = tokens[:truncate]
    if feat_delim is not None:
        tokens = [t.split(feat_delim)[layer] for t in tokens]
    return tokens


class TextMultiField(RawField):
    """Container for subfields.

    Text data might use POS/NER/etc labels in addition to tokens.
    This class associates the "base" :class:`Field` with any subfields.
    It also handles padding the data and stacking it.

    Args:
        base_name (str): Name for the base field.
        base_field (Field): The token field.
        feats_fields (Iterable[Tuple[str, Field]]): A list of name-field
            pairs.

    Attributes:
        fields (Iterable[Tuple[str, Field]]): A list of name-field pairs.
            The order is defined as the base field first, then
            ``feats_fields`` in alphabetical order.
    """

    def __init__(self, base_name, base_field, feats_fields):
        super(TextMultiField, self).__init__()
        self.fields = [(base_name, base_field)]
        for name, ff in sorted(feats_fields, key=lambda kv: kv[0]):
            self.fields.append((name, ff))

    @classmethod
    def from_opts(cls, base_name, n_feats, include_lengths=False,
                  truncate=None, pad='<blank>', bos='<s>', eos='</s>'):
        """Create text fields.

        Args:
            base_name (str): Name associated with the field.
            n_feats (int): Number of word level feats (not counting the tokens)
            include_lengths (bool): Optionally return the sequence lengths.
            truncate (bool or NoneType, optional): Defaults to ``None``.
            pad (str, optional): Defaults to ``"<blank>"``.
            bos (str or NoneType, optional): Defaults to ``"<s>"``.
            eos (str or NoneType, optional): Defaults to ``"</s>"``.

        Returns:
            TextMultiField
        """

        fields_ = []
        feat_delim = u"￨" if n_feats > 0 else None
        for i in range(n_feats + 1):
            name = base_name + "_feat_" + str(i - 1) if i > 0 else base_name
            tokenize = partial(
                _feature_tokenize,
                layer=i,
                truncate=truncate,
                feat_delim=feat_delim)
            use_len = i == 0 and include_lengths
            feat = Field(
                init_token=bos, eos_token=eos,
                pad_token=pad, tokenize=tokenize,
                include_lengths=use_len)
            fields_.append((name, feat))
        assert fields_[0][0] == base_name  # sanity check
        return cls(fields_[0][0], fields_[0][1], fields_[1:])

    @property
    def base_field(self):
        return self.fields[0][1]

    def process(self, batch, device=None):
        """Convert outputs of preprocess into Tensors.

        Args:
            batch (List[List[List[str]]]): A list of length batch size.
                Each element is a list of the preprocess results for each
                field (which are lists of str "words" or feature tags.
            device (torch.device or str): The device on which the tensor(s)
                are built.

        Returns:
            torch.LongTensor or Tuple[LongTensor, LongTensor]:
                A tensor of shape ``(seq_len, batch_size, len(self.fields))``
                where the field features are ordered like ``self.fields``.
                If the base field returns lengths, these are also returned
                and have shape ``(batch_size,)``.
        """

        # batch (list(list(list))): batch_size x len(self.fields) x seq_len
        batch_by_feat = list(zip(*batch))
        base_data = self.base_field.process(batch_by_feat[0], device=device)
        if self.base_field.include_lengths:
            # lengths: batch_size
            base_data, lengths = base_data

        feats = [ff.process(batch_by_feat[i], device=device)
                 for i, (_, ff) in enumerate(self.fields[1:], 1)]
        levels = [base_data] + feats
        # data: seq_len x batch_size x len(self.fields)
        data = torch.stack(levels, 2)
        if self.base_field.include_lengths:
            return data, lengths
        else:
            return data

    def preprocess(self, x):
        """Preprocess data.

        Args:
            x (str): A sentence string (words joined by whitespace).

        Returns:
            List[List[str]]: A list of length ``len(self.fields)`` containing
                lists of tokens/feature tags for the sentence. The output
                is ordered like ``self.fields``.
        """

        return [f.preprocess(x) for _, f in self.fields]

    def __getitem__(self, item):
        return self.fields[item]


class DocTextMultiField(RawField):
    """Container for Document-level fields.

    Document examples composed by several line of src-tgt pair. Example build
    from this, should be stored as a document which have list of src/tgt.
    And will formulate multiply contexted src/tgt pair.

    """

    def __init__(self, base_tm_name, base_tm_field, ctxs_tm_fields):
        assert isinstance(base_tm_field, TextMultiField),\
            "DocTextField should only have TextMultiField instances."
        super(DocTextMultiField, self).__init__()
        self.tm_fields = [(base_tm_name, base_tm_field)]
        for tm_name, tm_ff in sorted(ctxs_tm_fields, key=lambda kv: kv[0]):
            assert isinstance(tm_ff, TextMultiField),\
                "DocTextField should only have TextMultiField instances."
            self.tm_fields.append((tm_name, tm_ff))
        self.fields = [ff for tm_name, tm_ff in self.tm_fields for ff in tm_ff]

    @classmethod
    def from_opts(cls, base_name, n_ctxs, n_feats, include_lengths=False,
                  truncate=None, pad='<blank>', bos='<s>', eos='</s>'):
        """Create text fields.

        Args:
            base_name (str): Name associated with the field.
            n_ctxs (int): Number of previous segment as contexts.
            n_feats (int): Number of word level feats (not counting the tokens)
            include_lengths (bool): Optionally return the sequence lengths.
            truncate (bool or NoneType, optional): Defaults to ``None``.
            pad (str, optional): Defaults to ``"<blank>"``.
            bos (str or NoneType, optional): Defaults to ``"<s>"``.
            eos (str or NoneType, optional): Defaults to ``"</s>"``.

        Returns:
            DocTextMultiField
        """

        assert n_ctxs > 0, "To build DocTextMultiField, n_ctxs > 0."

        fields_ = []
        for i in range(n_ctxs + 1):
            sub_name = base_name + "_ctx_" + str(i - 1) if i > 0 else base_name
            sub_field = TextMultiField.from_opts(
                sub_name, n_feats, include_lengths=include_lengths,
                truncate=truncate, pad=pad, bos=bos, eos=eos)
            fields_.append((sub_name, sub_field))
        assert fields_[0][0] == base_name  # sanity check

        return cls(fields_[0][0], fields_[0][1], fields_[1:])

    @property
    def base_tm_field(self):
        """Return base TextMultiField: the current src/tgt with features."""
        return self.tm_fields[0][1]

    @property
    def ctxs_tm_fields(self):
        """Return contexts TextMultiField: previous src/tgt with features."""
        return self.tm_fields[1:]

    @property
    def base_field(self):
        """Return true base Field: the current src/tgt."""
        return self.base_tm_field.base_field

    @property
    def base_fields(self):
        """Return all base Fields: the current src/tgt or contexts src/tgt."""
        return [tm_field.base_field for (_, tm_field) in self.tm_fields]

    def preprocess(self, ctxed_x):
        """Preprocess data to be stored in .pt file.

        Args:
            ctxed_x (tuple(str, list)): a tuple of (str, list[str]), the first
                element as src/tgt, second as list of contexts.

        Returns:
            tuple(List[List[str]], List[List[List[str]]]):
                1st element is base field
        """
        assert isinstance(ctxed_x, tuple),\
            'DocTextMultiField.preprocess should be passed with tuple.'
        base_x, ctxs = ctxed_x
        assert len(ctxs) == len(self.ctxs_tm_fields),\
            "number of contexts should be same as context fields"
        base_prep = [f.preprocess(base_x) for _, f in self.base_tm_field]
        ctxs_prep = [[f.preprocess(ctx) for _, f in tm_f]
                     for (_, tm_f), ctx in zip(self.ctxs_tm_fields, ctxs)]
        return (base_prep, ctxs_prep)

    def process(self, batch, device=None):
        """Convert output of preprocess into Tensor.

        Args:
            batch (List(tuple(List[List[str]], List[List[List[str]]]))):
                A batch(list) of Examples stored in .pt file.

        Returns:
            List[torch.LongTensor] or List[Tuple[LongTensor, LongTensor]]:
                listf of tensor in shape ``(seq_len, batch_size, n_feats)``.
                If the base field returns lengths, these are also returned
                and have shape ``(batch_size,)``.
        """
        # batch_base: List[List[List[str]]], index on (example)
        # batch_ctxs: List[List[List[List[str]]]], index on (example, n_ctxs)
        batch_base, batch_ctxs = list(zip(*batch))
        # batch_by_ctx: List[List[List[List[str]]]], index on (n_ctxs, example)
        batch_by_ctx = list(zip(*batch_ctxs))
        base_proc = self.base_tm_field.process(batch_base)
        ctxs_proc = [tm_f.process(ctx_b) for (_, tm_f), ctx_b in zip(
                     self.ctxs_tm_fields, batch_by_ctx)]
        batched_ctx_text = [base_proc] + ctxs_proc
        return batched_ctx_text

    def __getitem__(self, item):
        return self.fields[item]


def text_fields(**kwargs):
    """Create text document fields.

    Args:
        base_name (str): Name associated with the field.
        n_ctxs (int): Number of previous segment as contexts.
        n_feats (int): Number of word level feats (not counting the tokens)
        include_lengths (bool): Optionally return the sequence lengths.
        pad (str, optional): Defaults to ``"<blank>"``.
        bos (str or NoneType, optional): Defaults to ``"<s>"``.
        eos (str or NoneType, optional): Defaults to ``"</s>"``.
        truncate (bool or NoneType, optional): Defaults to ``None``.

    Returns:
        TextMultiField if n_ctxs=0 else DocTextMultiField
    """

    base_name = kwargs["base_name"]
    n_ctxs = kwargs.get("n_ctxs", 0)
    n_feats = kwargs["n_feats"]
    include_lengths = kwargs["include_lengths"]
    pad = kwargs.get("pad", "<blank>")
    bos = kwargs.get("bos", "<s>")
    eos = kwargs.get("eos", "</s>")
    truncate = kwargs.get("truncate", None)

    if n_ctxs == 0:
        field = TextMultiField.from_opts(
            base_name, n_feats, include_lengths=include_lengths,
            truncate=truncate, pad=pad, bos=bos, eos=eos
        )
    else:
        field = DocTextMultiField.from_opts(
            base_name, n_ctxs, n_feats, include_lengths=include_lengths,
            truncate=truncate, pad=pad, bos=bos, eos=eos
        )

    return field
