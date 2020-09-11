"""Parse cls for dynamic."""
from onmt.utils.parse import ArgumentParser
from onmt.utils.logging import logger


class DynamicArgumentParser(ArgumentParser):

    @classmethod
    def validate_dynamic_corpus(cls, opt):
        """Parse corpus specified in data field of YAML file."""
        import yaml
        default_transforms = opt.transforms
        if len(default_transforms) != 0:
            logger.info(f"Default transforms: {default_transforms}.")
        corpora = yaml.safe_load(opt.data)
        for cname, corpus in corpora.items():
            # Check path
            path_src = corpus.get('path_src', None)
            path_tgt = corpus.get('path_tgt', None)
            if path_src is None or path_tgt is None:
                raise ValueError(f'Corpus {cname} path are required')
            # Check language
            src_lang = corpus.get('src_lang', None)
            tgt_lang = corpus.get('tgt_lang', None)
            if src_lang is None or tgt_lang is None:
                raise ValueError(f'Corpus {cname} lang info are required.')
            # Check weight
            weight = corpus.get('weight', None)
            if weight is None:
                logger.warning(f"Corpus {cname}'s weight should be given."
                               " We default it to 1 for you.")
                corpus['weight'] = 1
            # Check Transforms
            _transforms = corpus.get('transforms', None)
            if _transforms is None:
                logger.info(f"Missing transforms field for {cname} data, "
                            f"set to default: {default_transforms}.")
                corpus['transforms'] = default_transforms
        logger.info(f"Parsed {len(corpora)} corpora from -data.")
        opt.data = corpora

    @classmethod
    def _validate_transforms(cls, opt):
        assert 0 <= opt.subword_alpha <= 1, \
            "subword_alpha should be in the range [0, 1]"
        kwargs_dict = eval(opt.onmttok_kwargs)
        if not isinstance(kwargs_dict, dict):
            raise ValueError(f"-onmttok_kwargs is not a dict valid string.")
        opt.onmttok_kwargs = kwargs_dict

    @classmethod
    def get_all_transform(cls, opt):
        """Should only called after `valid_dynamic_corpus`."""
        cls._validate_transforms(opt)
        all_transforms = set(opt.transforms)
        for cname, corpus in opt.data.items():
            _transforms = set(corpus['transforms'])
            if len(_transforms) != 0:
                all_transforms.update(_transforms)
        opt._all_transform = all_transforms
