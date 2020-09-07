#!/usr/bin/env python
from onmt.utils.logging import init_logger

from onmt.dynamic.parse import DynamicArgumentParser
from onmt.dynamic.opts import dynamic_preprocess_opts
from onmt.dynamic.corpus import save_transformed_sample
from onmt.dynamic.vocab import build_dynamic_fields, save_fields
from onmt.dynamic.transform import make_transforms, save_transforms, \
    get_specials, get_transforms_cls

# from collections import defaultdict
from collections import Counter

def build_vocab_main(opts):
    DynamicArgumentParser.validate_dynamic_corpus(opts)
    DynamicArgumentParser.get_all_transform(opts)
    init_logger()

    transforms_cls = get_transforms_cls(opts._all_transform)
    fields = None

    transforms = make_transforms(opts, transforms_cls, fields)
    counters_src, counters_tgt = save_transformed_sample(
        opts, transforms, n_sample=25000, build_vocab=True)
    print("Counters src:", len(counters_src))
    print("Counters tgt:", len(counters_tgt))
    if opts.share_vocab:
        counters_src += counters_tgt
        counters_tgt = counters_src

    if opts.dump_vocab is not None:
        with open(opts.dump_vocab + ".src", "w") as fs,\
                open(opts.dump_vocab + ".tgt", "w") as ft:
            for tok, count in counters_src.most_common():
                fs.write(tok + "\t" + str(count) + "\n")
            for tok, count in counters_tgt.most_common():
                ft.write(tok + "\t" + str(count) + "\n")

def get_parser():
    parser = DynamicArgumentParser(description='build_vocab.py')
    dynamic_preprocess_opts(parser)
    return parser


def main():
    parser = get_parser()
    opts, unknown = parser.parse_known_args()
    build_vocab_main(opts)


if __name__ == '__main__':
    main()
