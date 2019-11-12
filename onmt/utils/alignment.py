# -*- coding: utf-8 -*-

import torch
from itertools import accumulate
from collections import defaultdict
from re import finditer


def make_batch_align_matrix(index_tensor, size=None, normalize=False):
    """
    Convert a sparse index_tensor into a batch of alignment matrix,
    with row normalize to the sum of 1 if set normalize.

    Args:
        index_tensor (LongTensor): ``(N, 3)`` of [batch_id, tgt_id, src_id]
        size (List[int]): Size of the sparse tensor.
        normalize (bool): if normalize the 2nd dim of resulting tensor.
    """
    n_fill, device = index_tensor.size(0), index_tensor.device
    value_tensor = torch.ones([n_fill], dtype=torch.float)
    dense_tensor = torch.sparse_coo_tensor(
        index_tensor.t(), value_tensor, size=size, device=device).to_dense()
    if normalize:
        row_sum = dense_tensor.sum(-1, keepdim=True)  # sum by row(tgt)
        # threshold on 1 to avoid div by 0
        torch.nn.functional.threshold(row_sum, 1, 1, inplace=True)
        dense_tensor.div_(row_sum)
    return dense_tensor


def extract_alignment(align_matrix, tgt_mask, src_lens, n_best):
    """
    Extract a batched align_matrix into its src indice alignment lists,
    with tgt_mask to filter out invalid tgt position as EOS/PAD.
    BOS already excluded from tgt_mask in order to match prediction.

    Args:
        align_matrix (Tensor): ``(B, tgt_len, src_len)``,
            attention head normalized by Softmax(dim=-1)
        tgt_mask (BoolTensor): ``(B, tgt_len)``, True for EOS, PAD.
        src_lens (LongTensor): ``(B,)``, containing valid src length
        n_best (int): a value indicating number of parallel translation.
        * B: denote flattened batch as B = batch_size * n_best.

    Returns:
        alignments (List[List[FloatTensor]]): ``(batch_size, n_best,)``,
         containing valid alignment matrix for each translation.
    """
    batch_size_n_best = align_matrix.size(0)
    assert batch_size_n_best % n_best == 0

    alignments = [[] for _ in range(batch_size_n_best // n_best)]

    # treat alignment matrix one by one as each have different lengths
    for i, (am_b, tgt_mask_b, src_len) in enumerate(
            zip(align_matrix, tgt_mask, src_lens)):
        valid_tgt = ~tgt_mask_b
        valid_tgt_len = valid_tgt.sum()
        # get valid alignment (sub-matrix from full paded aligment matrix)
        am_valid_tgt = am_b.masked_select(valid_tgt.unsqueeze(-1)) \
                           .view(valid_tgt_len, -1)
        valid_alignment = am_valid_tgt[:, :src_len]  # only keep valid src
        alignments[i // n_best].append(valid_alignment)

    return alignments


def build_align_pharaoh(valid_alignment):
    """Convert valid alignment matrix to i-j Pharaoh format.(0 indexed)"""
    align_pairs = []
    tgt_align_src_id = valid_alignment.argmax(dim=-1)

    for tgt_id, src_id in enumerate(tgt_align_src_id.tolist()):
        align_pairs.append(str(src_id) + "-" + str(tgt_id))
    align_pairs.sort(key=lambda x: x[-1])  # sort by tgt_id
    align_pairs.sort(key=lambda x: x[0])  # sort by src_id
    return align_pairs


def to_word_align(src, tgt, subword_align, mode):
    """Convert subword alignment to word alignment.

    Args:
        src (string): tokenized sentence in source language.
        tgt (string): tokenized sentence in target language.
        subword_align (string): align_pharaoh correspond to src-tgt.
        mode (string): tokenization mode used by src and tgt,
            choose from ["joiner", "spacer"].

    Returns:
        word_align (string): converted alignments correspand to
            detokenized src-tgt.
    """
    src, tgt = src.strip().split(), tgt.strip().split()
    subword_align = {(int(a), int(b)) for a, b in (x.split("-")
                     for x in subword_align.split())}
    if mode == 'joiner':
        src_map = subword_map_by_joiner(src, marker='￭')
        tgt_map = subword_map_by_joiner(tgt, marker='￭')
    elif mode == 'spacer':
        src_map = subword_map_by_spacer(src, marker='▁')
        tgt_map = subword_map_by_spacer(tgt, marker='▁')
    else:
        raise ValueError("Invalid value for argument mode!")
    word_align = list({"{}-{}".format(src_map[a], tgt_map[b])
                       for a, b in subword_align})
    word_align.sort(key=lambda x: x[-1])  # sort by tgt_id
    word_align.sort(key=lambda x: x[0])  # sort by src_id
    return " ".join(word_align)


def subword_map_by_joiner(subwords, marker='￭'):
    """Return word id for each subword token (annotate by joiner)."""
    flags = [0] * len(subwords)
    for i, tok in enumerate(subwords):
        if tok.endswith(marker):
            flags[i] = 1
        if tok.startswith(marker):
            assert i >= 1 and flags[i-1] != 1, \
                "Sentence `{}` not correct!".format(" ".join(subwords))
            flags[i-1] = 1
    marker_acc = list(accumulate([0] + flags[:-1]))
    word_group = [(i - maker_sofar) for i, maker_sofar
                  in enumerate(marker_acc)]
    return word_group


def subword_map_by_spacer(subwords, marker='▁'):
    """Return word id for each subword token (annotate by spacer)."""
    word_group = list(accumulate([int(marker in x) for x in subwords]))
    if word_group[0] == 1:  # when dummy prefix is set
        word_group = [item - 1 for item in word_group]
    return word_group


def check_consecutive(numbers):
    """Return if the argument is a list contains consecutive numbers."""
    numbers = sorted(numbers)
    min_number = numbers[0]
    consecutive_list = list(range(min_number, min_number + len(numbers)))
    return numbers == consecutive_list


def cover_translation(src, tgt, align, reserve_dict):
    """Given src/tgt sentence pair, for each src_token in reserve_dict,
    replaced its aligned tgt_token with reserve_dict[src_token]."""
    reserve_dict = {k.strip(): v.strip() for k, v in reserve_dict.items()}
    src_list, tgt_list = src.split(' '), tgt.split(' ')
    # 1. Get begin_index_str to token_index_list mapping dict
    src_tok_lens = [len(tok) for tok in src_list]
    src_tok_lens_space = map(lambda x: x + 1, src_tok_lens)
    src_tok_idxs = [0] + list(accumulate(src_tok_lens_space))[:-1]
    str_id2list_id = dict(zip(src_tok_idxs, range(len(src_tok_idxs))))
    # 2. get align_dict(key=src-id, value=tgt-id)
    align_dict = defaultdict(list)
    for x in align.split():
        a, b = x.split("-")
        align_dict[int(a)].append(int(b))

    tgt_replace_dict = {}  # save {'tgt': 'gold'} mapping
    replace_tok_id = []  # to avoid multiply replace

    # 3. search substring that need to cover
    reserve_keys = reserve_dict.keys()
    for reserve_key in reserve_keys:
        for match in finditer(reserve_key, src):
            find_str_idx = match.start()
            # 4. reserve_key exist in source string
            gold_tgt = reserve_dict[reserve_key]

            begin_id = str_id2list_id[find_str_idx]
            src_ids = _search_str_from_tok_list(
                src_list, reserve_key, begin_id=begin_id)

            if src_ids is not None:
                # 5. get tgt_ids that aligned to src_ids
                tgt_id_lists = [align_dict[src_id] for src_id in src_ids]
                tgt_ids = {val for tgt_id_list in tgt_id_lists
                           for val in tgt_id_list}
                tgt_ids = sorted(list(tgt_ids))
                if len(tgt_ids) > 0:
                    intersection = [candidate for candidate in tgt_ids
                                    if candidate in replace_tok_id]
                    if check_consecutive(tgt_ids) and len(intersection) == 0:
                        # 6. add valid tgt substring to be replace
                        replace_tok_id.extend(tgt_ids)
                        tgt_substr = ' '.join([
                            tgt_list[tok_id] for tok_id in tgt_ids
                        ])
                        tgt_replace_dict[tgt_substr] = gold_tgt
    if len(tgt_replace_dict) > 0:
        for tgt_substr, gold_tgt in tgt_replace_dict.items():
            tgt = tgt.replace(tgt_substr, gold_tgt)
    return tgt


def _search_str_from_tok_list(token_list, target, begin_id=0):
    """Search `target` token from token_list from index `begin_id`."""
    candidate_ids = None
    for current_id in range(begin_id, len(token_list)):
        candidate = ' '.join(token_list[begin_id: current_id + 1])
        if candidate == target:
            candidate_ids = list(range(begin_id, current_id + 1))
            break
    return candidate_ids


def spacer2joiner(sequence, joiner='￭'):
    """convert spacer annotated tokenized string to joiner annotation.

    NOTE: We assume the space marker are prefix as in pyonmttok
    or sentencepiece by default.
    """
    toks = sequence.split(' ')
    for i in range(len(toks)):
        if i != 0 and not toks[i].startswith('▁'):
            toks[i] = joiner + toks[i]
    newstr = ' '.join(toks).replace('▁', '')
    return newstr


def detok_on_joiner(sequence, joiner='￭'):
    """Detokenize tokenized sentence annotated by joiner."""
    return sequence.replace(joiner + ' ', '').replace(' ' + joiner, '')
