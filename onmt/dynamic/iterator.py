"""Module that contain iterator used for dynamic data."""
from itertools import cycle

from torchtext.data import Dataset as TorchtextDataset, \
    Example as TorchtextExample, batch as torchtext_batch
from onmt.inputters import str2sortkey
from onmt.inputters.inputter import max_tok_len, OrderedIterator

from onmt.dynamic.corpus import get_corpora, build_corpora_iters
from onmt.dynamic.transforms import load_transforms


class DatasetAdapter(object):
    """Adapte a buckets of tuples into examples of a torchtext Dataset."""

    valid_field_name = ('src', 'tgt', 'indices')

    def __init__(self, fields):
        self._valid_fields(fields)

    def _valid_fields(self, fields):
        valid_fields = []
        for f_name in self.valid_field_name:
            valid_fields.append((f_name, fields[f_name]))
        self.fields_list = valid_fields

    @staticmethod
    def _process(fields_list, item):
        src, tgt, transform, cid, index = item
        # this is a hack: appears quicker to apply it here
        # than in the ParallelCorpusIterator
        maybe_item = transform.apply(src, tgt, corpus_name=cid)
        if maybe_item is not None:
            src, tgt = maybe_item
        else:
            return None
        src_line = ' '.join(src)
        tgt_line = ' '.join(tgt)
        ex = TorchtextExample.fromlist(
                (src_line, tgt_line, index), fields_list)
        return ex

    def _to_examples(self, bucket):
        examples = []
        for item in bucket:
            ex = self._process(self.fields_list, item)
            if ex is not None:
                examples.append(ex)
        return examples

    def __call__(self, bucket):
        examples = self._to_examples(bucket)
        dataset = TorchtextDataset(examples, self.fields_list)
        return dataset


class MixingStrategy(object):
    """Mixing strategy that should be used in Data Iterator."""

    def __init__(self, iterables, weights):
        """Initilize neccessary attr."""
        self._valid_iterable(iterables, weights)
        self.iterables = iterables
        self.weights = weights

    def _valid_iterable(self, iterables, weights):
        iter_keys = iterables.keys()
        weight_keys = weights.keys()
        if iter_keys != weight_keys:
            raise ValueError(
                f"keys in {iterables} & {iterables} should be equal.")

    def __iter__(self):
        raise NotImplementedError


class SequentialMixer(MixingStrategy):
    """Generate data sequentially from `iterables` which is exhaustible."""

    def _iter_datasets(self):
        for ds_name, ds_weight in self.weights.items():
            for _ in range(ds_weight):
                yield ds_name

    def __iter__(self):
        for ds_name in self._iter_datasets():
            iterable = self.iterables[ds_name]
            yield from iterable


class WeightedMixer(MixingStrategy):
    """A mixing strategy that mix data weightedly and iterate infinitely."""

    def __init__(self, iterables, weights):
        super().__init__(iterables, weights)
        self._iterators = {
            ds_name: iter(generator)
            for ds_name, generator in self.iterables.items()
        }

    def _reset_iter(self, ds_name):
        self._iterators[ds_name] = iter(self.iterables[ds_name])

    def _iter_datasets(self):
        for ds_name, ds_weight in self.weights.items():
            for _ in range(ds_weight):
                yield ds_name

    def __iter__(self):
        for ds_name in cycle(self._iter_datasets()):
            iterator = self._iterators[ds_name]
            try:
                item = next(iterator)
            except StopIteration:
                self._reset_iter(ds_name)
                iterator = self._iterators[ds_name]
                item = next(iterator)
            finally:
                yield item


class DynamicDatasetIter(object):
    """Yield data from multiple parallel plain text corpora."""

    def __init__(self, corpora, transforms, fields, opts, is_train,
                 stride=1, offset=0):
        self.corpora = corpora
        self.transforms = transforms
        self.fields = fields
        self.corpora_info = opts.data
        self.is_train = is_train
        self.init_iterators = False
        self.batch_size = opts.batch_size if is_train \
            else opts.valid_batch_size
        self.batch_size_fn = max_tok_len \
            if is_train and opts.batch_type == "tokens" else None
        if opts.batch_size_multiple is not None:
            self.batch_size_multiple = opts.batch_size_multiple
        else:
            self.batch_size_multiple = 8 if opts.model_dtype == "fp16" else 1
        self.device = 'cpu'
        self.sort_key = str2sortkey[opts.data_type]
        self.bucket_size = opts.bucket_size
        self.pool_factor = opts.pool_factor
        self.stride = stride
        self.offset = offset

    def _init_datasets(self):
        datasets_iterables = build_corpora_iters(
            self.corpora, self.transforms,
            self.corpora_info, self.is_train,
            stride=self.stride, offset=self.offset)
        self.dataset_adapter = DatasetAdapter(self.fields)
        datasets_weights = {
            ds_name: int(self.corpora_info[ds_name]['weight'])
            for ds_name in datasets_iterables.keys()
        }
        if self.is_train:
            self.mixer = WeightedMixer(datasets_iterables, datasets_weights)
        else:
            self.mixer = SequentialMixer(datasets_iterables, datasets_weights)
        self.init_iterators = True

    def _bucketing(self):
        buckets = torchtext_batch(
            self.mixer,
            batch_size=self.bucket_size,
            batch_size_fn=None)
        yield from buckets

    def __iter__(self):
        if self.init_iterators is False:
            self._init_datasets()
        for bucket in self._bucketing():
            dataset = self.dataset_adapter(bucket)
            train_iter = OrderedIterator(
                dataset,
                self.batch_size,
                pool_factor=self.pool_factor,
                batch_size_fn=self.batch_size_fn,
                batch_size_multiple=self.batch_size_multiple,
                device=self.device,
                train=self.is_train,
                sort=False,
                sort_within_batch=True,
                sort_key=self.sort_key,
                repeat=False,
            )
            for batch in train_iter:
                yield batch


def build_dynamic_dataset_iter(fields, opts, is_train=True,
                               stride=1, offset=0):
    """Build `DynamicDatasetIter` from fields & opts."""
    transforms = load_transforms(opts)
    corpora = get_corpora(opts, is_train)
    if corpora is None:
        assert not is_train, "only valid corpus is ignorable."
        return None
    return DynamicDatasetIter(
        corpora, transforms, fields, opts, is_train,
        stride=stride, offset=offset)
