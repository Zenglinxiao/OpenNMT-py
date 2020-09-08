"""Module that contain shard utils for dynamic data."""
import os
from onmt.utils.logging import logger
from onmt.dynamic.transforms import TransformPipe

from collections import Counter

VALID_CORPUS_NAME = 'valid'


class ParallelCorpus(object):
    """A parallel corpus file pair that can be loaded to iterate."""

    def __init__(self, src, tgt):
        """Initialize src & tgt side file path."""
        self.src = src
        self.tgt = tgt

    def load(self, offset=0, stride=1):
        """
        Load file and iterate by lines.
        `offset` and `stride` allow to iterate only on every
        `stride` example, starting from `offset`.
        """
        import codecs
        with codecs.open(self.src, mode='rb') as fs,\
                codecs.open(self.tgt, mode='rb') as ft:
            logger.info(f"Loading {repr(self)}...")
            for i, (sline, tline) in enumerate(zip(fs, ft)):
                if (i % stride) == offset:
                    sline = sline.decode('utf-8')
                    tline = tline.decode('utf-8')
                    yield (sline, tline)

    def __repr__(self):
        cls_name = type(self).__name__
        return '{}({}, {})'.format(cls_name, self.src, self.tgt)


def get_corpora(opts, is_train=False):
    corpora_dict = {}
    if is_train:
        for corpus_id, corpus_dict in opts.data.items():
            if corpus_id != VALID_CORPUS_NAME:
                corpora_dict[corpus_id] = ParallelCorpus(
                    corpus_dict["path_src"],
                    corpus_dict["path_tgt"])
    else:
        if VALID_CORPUS_NAME in opts.data.keys():
            corpora_dict[VALID_CORPUS_NAME] = ParallelCorpus(
                opts.data[VALID_CORPUS_NAME]["path_src"],
                opts.data[VALID_CORPUS_NAME]["path_tgt"])
        else:
            return None
    return corpora_dict


class ParallelCorpusIterator(object):
    def __init__(self, cid, corpus, transform, infinitely=False,
                 stride=1, offset=0):
        self.cid = cid
        self.corpus = corpus
        self.transform = transform
        self.infinitely = infinitely
        self.stride = stride
        self.offset = offset

    def _tokenize(self, stream):
        for (sline, tline) in stream:
            sline = sline.strip('\n').split()
            tline = tline.strip('\n').split()
            yield (sline, tline)

    def _transform(self, stream):
        for item in stream:
            # item = self.transform.apply(*item, corpus_name=self.cid)
            item = (*item, self.transform, self.cid)
            if item is not None:
                yield item
        report_msg = self.transform.stats()
        if report_msg != '':
            logger.info("Transform statistics for {}:\n{}".format(
                self.cid, report_msg))

    def _add_index(self, stream, stride=1, offset=0):
        for i, item in enumerate(stream):
            yield (*item, i * stride + offset)

    def _iter_corpus(self):
        corpus_stream = self.corpus.load(stride=self.stride, offset=self.offset)
        tokenized_corpus = self._tokenize(corpus_stream)
        transformed_corpus = self._transform(tokenized_corpus)
        indexed_corpus = self._add_index(transformed_corpus, stride=self.stride, offset=self.offset)
        yield from indexed_corpus

    def __iter__(self):
        if self.infinitely:
            while True:
                _iter = self._iter_corpus()
                yield from _iter
        else:
            yield from self._iter_corpus()


def build_corpora_iters(corpora, transforms, corpora_info, train=False,
                        stride=1, offset=0):
    """Return `ParallelCorpusIterator` for all corpora defined in opts."""
    corpora_iters = dict()
    for c_id, corpus in corpora.items():
        c_transform_names = corpora_info[c_id].get('transforms', [])
        corpus_transform = [transforms[name] for name in c_transform_names]
        transform_pipe = TransformPipe.build_from(corpus_transform)
        logger.info(f"{c_id}'s transforms: {str(transform_pipe)}")
        corpus_iter = ParallelCorpusIterator(
            c_id, corpus, transform_pipe, infinitely=train,
            stride=stride, offset=offset)
        corpora_iters[c_id] = corpus_iter
    return corpora_iters


def save_transformed_sample(opts, transforms, n_sample=3, build_vocab=False):
    """Save transformed data sample as specified in opts."""
    corpora = get_corpora(opts, is_train=True)
    if build_vocab:
        counter_src = Counter()
        counter_tgt = Counter()
    datasets_iterables = build_corpora_iters(
        corpora, transforms,
        opts.data, train=True)
    sample_path = os.path.join(
        os.path.dirname(opts.save_data), 'sample')
    os.makedirs(sample_path, exist_ok=True)
    for c_name, c_iter in datasets_iterables.items():
        dest_base = os.path.join(sample_path, "{}.sample".format(c_name))
        with open(dest_base + ".src", 'w', encoding="utf-8") as f_src,\
                open(dest_base + ".tgt", 'w', encoding="utf-8") as f_tgt:
            for i, example in enumerate(c_iter):
                src, tgt, transform, cid, index = example
                maybe_item = transform.apply(src, tgt, corpus_name=cid)
                if maybe_item is None:
                    continue
                src, tgt = maybe_item
                # example = [src, tgt, index]
                if build_vocab:
                    counter_src.update(src)
                    counter_tgt.update(tgt)
                f_src.write(" ".join(src) + '\n')
                f_tgt.write(" ".join(tgt) + '\n')
                if i > n_sample:
                    break
    if build_vocab:
        return counter_src, counter_tgt
