#!/usr/bin/env python
import argparse
import random

from onmt.dynamicdata.config import read_data_config, sharding_only, \
    save_shard_config, verify_shard_config, normalize_sizes
from onmt.dynamicdata.shard import DataSharder
from onmt.dynamicdata.transforms import make_transform_models, \
    make_transforms, get_specials
from onmt.dynamicdata.vocab import load_vocabulary, prepare_fields, \
    save_fields, save_transforms, SimpleSharedVocabulary


def shard_main(config_file):
    data_config = read_data_config(config_file)
    data_config = sharding_only(data_config)
    normalize_sizes(data_config)
    save_shard_config(data_config)
    if data_config['meta']['shard']['share_vocab']:
        vocab = SimpleSharedVocabulary(data_config)
    else:
        raise NotImplementedError()
    # 0.1 of old prep default shard size
    max_shard_size = data_config['meta']['shard'].get('shard_size', 100000)
    max_initial_shards = data_config['meta']['shard'].get('initial_shards', 10)
    compress = data_config['meta']['shard'].get('compress', False)
    pretokenize = data_config['meta']['shard'].get('pretokenize', False)
    predetokenize = data_config['meta']['shard'].get('predetokenize', False)
    if pretokenize and predetokenize:
        raise Exception('Cannot both pretokenize and predetokenize')
    pre = None
    pre = 'tokenize' if pretokenize else pre
    pre = 'detokenize' if predetokenize else pre
    data_sharder = DataSharder(
        data_config,
        max_shard_size=max_shard_size,
        max_initial_shards=max_initial_shards,
        compress=compress,
        pre=pre,
        vocab_counter=vocab,
        )
    data_sharder()
    vocab.save_all()


def vocab_main(config_file):
    data_config = read_data_config(config_file)
    verify_shard_config(data_config)
    vocabs = load_vocabulary(data_config)
    transform_models = make_transform_models(data_config, vocabs)
    transforms = make_transforms(transform_models, data_config, vocabs)
    fields = prepare_fields(data_config, vocabs, get_specials(transforms))
    save_fields(data_config, fields)
    save_transforms(data_config, transform_models, transforms)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('subcommand', choices=['shard', 'vocab'])
    parser.add_argument('config')
    parser.add_argument('--seed', default=1)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    random.seed(args.seed)
    if args.subcommand == 'shard':
        shard_main(args.config)
    elif args.subcommand == 'vocab':
        vocab_main(args.config)


if __name__ == '__main__':
    main()
