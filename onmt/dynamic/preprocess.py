#!/usr/bin/env python
from onmt.utils.logging import init_logger

from onmt.dynamic.parse import DynamicArgumentParser
from onmt.dynamic.opts import dynamic_preprocess_opts
from onmt.dynamic.corpus import save_dataset, save_transformed_sample
from onmt.dynamic.vocab import build_dynamic_fields, save_fields
from onmt.dynamic.transforms import make_transforms, save_transforms, \
    get_specials, get_transforms_cls


def preprocess_main(opts):
    DynamicArgumentParser.validate_dynamic_corpus(opts)
    DynamicArgumentParser.get_all_transform(opts)
    init_logger()

    transforms_cls = get_transforms_cls(opts._all_transform)
    specials = get_specials(opts, transforms_cls)

    fields = build_dynamic_fields(
        opts, src_specials=specials['src'], tgt_specials=specials['tgt'])
    save_fields(opts, fields)

    transforms = make_transforms(opts, transforms_cls, fields)
    save_transforms(opts, transforms)
    # Gather corpus to preprocess directory
    save_dataset(opts.data, opts.save_data, opts.overwrite)
    if opts.verbose:
        save_transformed_sample(opts, transforms)


def get_parser():
    parser = DynamicArgumentParser(description='dynamic_preprocess.py')
    dynamic_preprocess_opts(parser)
    return parser


def main():
    parser = get_parser()
    opts, unknown = parser.parse_known_args()
    preprocess_main(opts)


if __name__ == '__main__':
    main()
