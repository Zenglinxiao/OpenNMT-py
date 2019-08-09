"""
This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

import onmt
from onmt.modules.sparse_losses import SparsemaxLoss
from onmt.modules.sparse_activations import LogSparsemax
from sklearn.metrics import f1_score


def build_loss_compute(model, tgt_field, opt, train=True):
    """
    Returns a LossCompute subclass which wraps around an nn.Module subclass
    (such as nn.NLLLoss) which defines the loss criterion. The LossCompute
    object allows this loss to be computed in shards and passes the relevant
    data to a Statistics object which handles training/validation logging.
    Currently, the NMTLossCompute class handles all loss computation except
    for when using a copy mechanism.
    """
    device = torch.device("cuda" if onmt.utils.misc.use_gpu(opt) else "cpu")
    if opt.is_bert is True:
        assert hasattr(model, 'bert')
        if tgt_field.pad_token is not None:
            if tgt_field.use_vocab:
                padding_idx = tgt_field.vocab.stoi[tgt_field.pad_token]
            else:  # target is pre-numerized: -1 for unmasked token in mlm
                padding_idx = tgt_field.pad_token
            criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='mean')
        else:  # sentence level
            criterion = nn.NLLLoss(reduction='mean')
        task = opt.task_type
        compute = BertLoss(criterion, task)
    else:
        assert isinstance(model, onmt.models.NMTModel)
        padding_idx = tgt_field.vocab.stoi[tgt_field.pad_token]
        unk_idx = tgt_field.vocab.stoi[tgt_field.unk_token]
        if opt.copy_attn:
            criterion = onmt.modules.CopyGeneratorLoss(
                len(tgt_field.vocab), opt.copy_attn_force,
                unk_index=unk_idx, ignore_index=padding_idx
            )
        elif opt.label_smoothing > 0 and train:
            criterion = LabelSmoothingLoss(
                opt.label_smoothing, len(tgt_field.vocab), ignore_index=padding_idx
            )
        elif isinstance(model.generator[-1], LogSparsemax):
            criterion = SparsemaxLoss(ignore_index=padding_idx, reduction='sum')
        else:
            criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')

        # if the loss function operates on vectors of raw logits instead of
        # probabilities, only the first part of the generator needs to be
        # passed to the NMTLossCompute. At the moment, the only supported
        # loss function of this kind is the sparsemax loss.
        use_raw_logits = isinstance(criterion, SparsemaxLoss)
        loss_gen = model.generator[0] if use_raw_logits else model.generator
        if opt.copy_attn:
            compute = onmt.modules.CopyGeneratorLossCompute(
                criterion, loss_gen, tgt_field.vocab, opt.copy_loss_by_seqlength
            )
        else:
            compute = NMTLossCompute(criterion, loss_gen)
    compute.to(device)
    return compute


class BertLoss(nn.Module):
    def __init__(self, criterion, task):
        super(BertLoss, self).__init__()
        self.criterion = criterion
        self.task = task

    @property
    def padding_idx(self):
        return self.criterion.ignore_index

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _stats(self, loss, tokens_scores, tokens_target,
               sents_scores, sents_target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            tokens_scores (:obj:`FloatTensor`): scores for each token
            tokens_target (:obj:`FloatTensor`): true targets for each token
            sents_scores (:obj:`FloatTensor`): scores for each sentence
            sents_target (:obj:`FloatTensor`): true targets for each sentence

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        if self.task == 'pretraining':
            # masked lm task: token level
            pred_tokens = tokens_scores.argmax(1)  # (B*S, V) -> (B*S)
            non_padding = tokens_target.ne(self.padding_idx)  # mask: (B*S)
            tokens_match = pred_tokens.eq(tokens_target).masked_select(non_padding)
            n_correct_tokens = tokens_match.sum().item()
            n_tokens = non_padding.sum().item()
            f1 = 0
            # next sentence prediction task: sentence level
            pred_sents = sents_scores.argmax(-1)  # (B, 2) -> (2)
            n_correct_sents = sents_target.eq(pred_sents).sum().item()
            n_sentences = len(sents_target)

        elif self.task == 'classification':
            # token level task: Not valide
            n_correct_tokens = 0
            n_tokens = 0
            f1 = 0
            # sentence level task:
            pred_sents = sents_scores.argmax(-1)  # (B, n_label) -> (n_label)
            n_correct_sents = sents_target.eq(pred_sents).sum().item()
            n_sentences = len(sents_target)

        elif self.task == 'tagging':
            # token level task:
            pred_tokens = tokens_scores.argmax(1)  # (B*S, V) -> (B*S)
            non_padding = tokens_target.ne(self.padding_idx)  # mask: (B*S)
            tokens_match = pred_tokens.eq(tokens_target).masked_select(non_padding)
            n_correct_tokens = tokens_match.sum().item()
            n_tokens = non_padding.sum().item()
            # for f1:
            tokens_target_select = tokens_target.masked_select(non_padding)
            pred_tokens_select = pred_tokens.masked_select(non_padding)
            f1 = f1_score(tokens_target_select.cpu(),
                          pred_tokens_select.cpu(), average="micro")

            # sentence level task: Not valide
            n_correct_sents = 0
            n_sentences = 0

        elif self.task == 'generation':
            # token level task:
            pred_tokens = tokens_scores.argmax(1)  # (B*S, V) -> (B*S)
            non_padding = tokens_target.ne(self.padding_idx)  # mask: (B*S)
            tokens_match = pred_tokens.eq(tokens_target).masked_select(non_padding)
            n_correct_tokens = tokens_match.sum().item()
            n_tokens = non_padding.sum().item()
            f1 = 0
            # sentence level task: Not valide
            n_correct_sents = 0
            n_sentences = 0
        else:
            raise ValueError("task %s not available!" % (self.task))

        return onmt.utils.BertStatistics(loss.item(), n_tokens,
                                         n_correct_tokens, n_sentences,
                                         n_correct_sents, f1)


    def forward(self, batch, outputs):
        """
        Args:
            batch: batch of examples
            outputs: tuple of log proba for next sentense & lm
                (seq_class_log_prob:(batch, 2),
                prediction_log_prob:(batch, seq, vocab))
        """
        assert isinstance(outputs, tuple)
        seq_class_log_prob, prediction_log_prob = outputs
        if self.task == 'pretraining':
            assert list(seq_class_log_prob.size()) == [len(batch), 2]
            # masked lm task: token level(loss mean by number of tokens)
            gtruth_tokens = batch.lm_labels_ids  # (B, S)
            bottled_gtruth_tokens = gtruth_tokens.view(-1)  # (B, S)
            # prediction: (B, S, V) -> (B * S, V)
            bottled_prediction_log_prob = self._bottle(prediction_log_prob)
            mask_loss = self.criterion(bottled_prediction_log_prob,
                                       bottled_gtruth_tokens)
            # next sentence prediction task: sentence level(mean by sentence)
            gtruth_sentences = batch.is_next  # (B,)
            next_loss = self.criterion(seq_class_log_prob, gtruth_sentences)
            total_loss = next_loss + mask_loss  # total_loss reduced by mean

        elif self.task == 'classification':
            assert prediction_log_prob is None
            assert hasattr(batch, 'category')
            # token level task: Not valide
            bottled_prediction_log_prob = None
            bottled_gtruth_tokens = None
            # sentence level task: loss mean by number of sentences
            gtruth_sentences = batch.category
            total_loss = self.criterion(seq_class_log_prob, gtruth_sentences)

        elif self.task == 'tagging' or self.task == 'generation':
            assert seq_class_log_prob is None
            assert hasattr(batch, 'token_labels')
            # token level task: loss mean by number of tokens
            gtruth_tokens = batch.token_labels  # (B, S)
            bottled_gtruth_tokens = gtruth_tokens.view(-1)  # (B, S)
            # prediction: (B, S, V) -> (B * S, V)
            bottled_prediction_log_prob = self._bottle(prediction_log_prob)
            total_loss = self.criterion(bottled_prediction_log_prob,
                                        bottled_gtruth_tokens)
            # sentence level task: Not valide
            seq_class_log_prob = None
            gtruth_sentences = None

        else:
            raise ValueError("task %s not available!" % (self.task))

        stats = self._stats(total_loss.clone(),
                            bottled_prediction_log_prob,
                            bottled_gtruth_tokens,
                            seq_class_log_prob,
                            gtruth_sentences)
        return total_loss, stats


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating multiple
    loss computations

    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, criterion, generator):
        super(LossComputeBase, self).__init__()
        self.criterion = criterion
        self.generator = generator

    @property
    def padding_idx(self):
        return self.criterion.ignore_index

    def _make_shard_state(self, batch, output, range_, attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def __call__(self,
                 batch,
                 output,
                 attns,
                 normalization=1.0,
                 shard_size=0,
                 trunc_start=0,
                 trunc_size=None):
        """Compute the forward loss, possibly in shards in which case this
        method also runs the backward pass and returns ``None`` as the loss
        value.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(trunc_start, trunc_start + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          normalization: Optional normalization factor.
          shard_size (int) : maximum number of examples in a shard
          trunc_start (int) : starting position of truncation window
          trunc_size (int) : length of truncation window

        Returns:
            A tuple with the loss and a :obj:`onmt.utils.Statistics` instance.
        """
        if trunc_size is None:
            trunc_size = batch.tgt.size(0) - trunc_start
        trunc_range = (trunc_start, trunc_start + trunc_size)
        shard_state = self._make_shard_state(batch, output, trunc_range, attns)
        if shard_size == 0:
            loss, stats = self._compute_loss(batch, **shard_state)
            return loss / float(normalization), stats
        batch_stats = onmt.utils.Statistics()
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)
            loss.div(float(normalization)).backward()
            batch_stats.update(stats)
        return None, batch_stats

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()
        return onmt.utils.Statistics(loss.item(), num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, criterion, generator, normalization="sents"):
        super(NMTLossCompute, self).__init__(criterion, generator)

    def _make_shard_state(self, batch, output, range_, attns=None):
        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1], :, 0],
        }

    def _compute_loss(self, batch, output, target):
        bottled_output = self._bottle(output)

        scores = self.generator(bottled_output)
        gtruth = target.view(-1)

        loss = self.criterion(scores, gtruth)
        stats = self._stats(loss.clone(), scores, gtruth)

        return loss, stats


def filter_shard_state(state, shard_size=None):
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
