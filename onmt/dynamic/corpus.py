"""Module that contain shard utils for dynamic data."""
import os
from onmt.utils.logging import logger
from onmt.dynamic.transforms import TransformPipe


VALID_CORPUS_NAME = 'valid'


def make_link(filepath, link, overwrite=False):
    """Make a soft link with name `link` to `filepath`."""
    root = os.getcwd()
    filepath = os.path.join(root, filepath)
    link = os.path.join(root, link)
    if not os.path.exists(filepath):
        raise OSError(f"File not found at {filepath}.")
    if os.path.exists(link):
        if not overwrite:
            logger.warning(f"Link {link} already exist. Skipping")
            return
        else:
            logger.warning(f"Link {link} exist. overwriting it.")
            os.remove(link)
    os.symlink(filepath, link)


def save_dataset(corpora, save_data, overwrite=False):
    """Save `corpora` to `save_data`.

    Args:
        corpora (list): a dictionary of corpus with name as keys;
        save_data (str): file prefix for saving;
        overwrite (bool): if overwrite existing corpus.
    """
    os.makedirs(os.path.dirname(save_data), exist_ok=True)
    for corpus_name, corpus_dict in corpora.items():
        if corpus_name == VALID_CORPUS_NAME:
            dest_base = "{}.{}".format(save_data, VALID_CORPUS_NAME)
        else:
            dest_base = "{}.train_{}".format(save_data, corpus_name)
        make_link(corpus_dict['path_src'], dest_base + '.src', overwrite)
        make_link(corpus_dict['path_tgt'], dest_base + '.tgt', overwrite)


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


def get_corpus_paths(base_dir, data_basename, corpus_name):
    if corpus_name != VALID_CORPUS_NAME:
        src = os.path.join(
            base_dir,
            "{}.train_{}.src".format(
                data_basename, corpus_name))
        tgt = os.path.join(
            base_dir,
            "{}.train_{}.tgt".format(
                data_basename, corpus_name))
    else:
        src = os.path.join(
            base_dir,
            "{}.{}.src".format(
                data_basename, corpus_name))
        tgt = os.path.join(
            base_dir,
            "{}.{}.tgt".format(
                data_basename, corpus_name))
    return ParallelCorpus(src, tgt)


def get_corpora(opts, is_train=False):
    base_dir = os.path.dirname(opts.save_data)
    data_basename = os.path.basename(opts.save_data)
    if is_train:
        corpora_ids = [c_id for c_id in opts.data.keys()
                       if c_id != VALID_CORPUS_NAME]
        if len(corpora_ids) == 0:
            raise ValueError("Please specify training corpus!")
    else:
        if VALID_CORPUS_NAME in opts.data.keys():
            corpora_ids = [VALID_CORPUS_NAME]
        else:
            return None
    corpora_dict = {}
    for corpus_id in corpora_ids:
        corpus = get_corpus_paths(
            base_dir, data_basename, corpus_id)
        corpora_dict[corpus_id] = corpus
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

    def _add_index(self, stream):
        for i, item in enumerate(stream):
            yield (*item, i)

    def _iter_corpus(self):
        corpus_stream = self.corpus.load(stride=self.stride, offset=self.offset)
        tokenized_corpus = self._tokenize(corpus_stream)
        transformed_corpus = self._transform(tokenized_corpus)
        indexed_corpus = self._add_index(transformed_corpus)
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


def save_transformed_sample(opts, transforms, n_sample=3):
    """Save transformed data sample as specified in opts."""
    corpora = get_corpora(opts, is_train=True)
    datasets_iterables = build_corpora_iters(
        corpora, transforms,
        opts.data, train=True)
    sample_path = os.path.join(
        os.path.dirname(opts.save_data), 'sample')
    os.makedirs(sample_path, exist_ok=True)
    for c_name, c_iter in datasets_iterables.items():
        dest_base = os.path.join(sample_path, "{}.sample".format(c_name))
        with open(dest_base, 'w', encoding="utf-8") as f_sample:
            for i, example in enumerate(c_iter):
                src, tgt, transform, cid, index = example
                maybe_item = transform.apply(src, tgt, corpus_name=cid)
                if maybe_item is not None:
                    src, tgt = maybe_item
                example = [src, tgt, index]
                item_list = []
                for item in example:
                    if isinstance(item, list):
                        item = ' '.join(item)
                    item_list.append(str(item))
                line = '\t'.join(item_list)
                f_sample.write(line + '\n')
                if i > n_sample:
                    break
