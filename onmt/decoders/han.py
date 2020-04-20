"""
 Implementation of "Document-Level Neural Machine Translation
 with Hierarchical Attention Networks"
"""

from onmt.decoders.transformer import TransformerDecoder
from onmt.modules.han_context import HierarquicalContext


class HANDecoder(TransformerDecoder):
    """The HAN decoder from "Document-Level Neural Machine Translation
    with Hierarchical Attention Networks".

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, num_layers, d_model, heads, d_ff,
                 copy_attn, self_attn_type, dropout, attention_dropout,
                 embeddings, max_relative_positions, aan_useffn,
                 full_context_alignment, alignment_layer,
                 alignment_heads, context_type, context_size, mode,
                 padding_idx, bos_idx=None, eos_idx=None):
        super(HANDecoder, self).__init__(
            num_layers, d_model, heads, d_ff, copy_attn, self_attn_type,
            dropout, attention_dropout, embeddings, max_relative_positions,
            aan_useffn, full_context_alignment,
            alignment_layer, alignment_heads)
        # context_type: han(HAN_dec/HAN_join), HAN_dec_context, HAN_dec_source
        raise NotImplementedError
        self.context_type = context_type
        self.contextualize = HierarquicalContext(
            d_model, heads, d_ff, dropout, attention_dropout, context_size,
            padding_idx=padding_idx, bos_idx=bos_idx, eos_idx=eos_idx)
        self._set_mode(mode)

    def _set_mode(self, mode):
        """Set the training mode."""
        assert mode in ['all', 'sent', 'context'],\
            f"{mode} is not supported!"
        self.mode = mode
        if self.mode == 'context':
            # only train 'context' part, freeze the rest
            for param in self.parameters():
                param.require_grad = False
            for param in self.contextualize.parameters():
                param.require_grad = True

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.copy_attn,
            opt.self_attn_type,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.attention_dropout,
            embeddings,
            max_relative_positions=opt.max_relative_positions,
            aan_useffn=opt.aan_useffn,
            full_context_alignment=opt.full_context_alignment,
            alignment_layer=opt.alignment_layer,
            alignment_heads=opt.alignment_heads,
            context_type=opt.decoder_type,
            context_size=opt.n_tgt_ctxs,
            mode=opt.decoder_mode,
            padding_idx=embeddings.word_padding_idx,
            bos_idx=embeddings.word_bos_idx,
            eos_idx=embeddings.word_eos_idx)

    def forward(self, tgt, memory_bank, step=None, ctxs=None, **kwargs):
        """Decode, possibly stepwise."""
        # 1. Transformer part forward
        # # TODO: to be done...
        # assert step is None, "decoder not support stepwise for instance."
        dec_outs, attns = super().forward(tgt, memory_bank, step, **kwargs)
        # 2. Context part
        if self.mode in ['all', 'context']:
            if self.context_type == "han":
                # use for HAN_dec/HAN_join
                dec_out_ctxs, attns_ctxs = self.ctxs_forward(
                    ctxs, memory_bank, step, **kwargs)
                dec_outs, attn_word_enc_l, attn_sent_enc = self.contextualize(
                    dec_outs, ctxs, dec_out_ctxs)
            else:
                # dec_out_ctxs(ctxs) may vray according to type
                # 1. HAN_dec_context: need to pass mid(of context_attn in dec)
                #  as dec_out_ctxs
                # 2. HAN_dec_source: need to pass HANEncoder out (memory_bank)
                #  as dec_out_ctxs, src_ctxs as ctxs for getting masks.
                raise ValueError(f"{self.context_type} not supported!")
        return dec_outs, attns

    def ctxs_forward(self, ctxs, memory_bank, step=None, **kwargs):
        """Base TransformerDecoder forward for all ctxs."""
        dec_out_ctxs, attns_ctxs = [], []
        if ctxs is not None:
            for ctx in ctxs:
                ctx, ctx_lengths = ctx if isinstance(ctx, tuple) \
                    else (ctx, None)
                out_ctx, attns_ctx = super().forward(
                    ctx, memory_bank, step, **kwargs)
                dec_out_ctxs.append(out_ctx)
                attns_ctxs.append(attns_ctx)
        return dec_out_ctxs, attns_ctxs

    def update_dropout(self, dropout, attention_dropout):
        super().update_dropout(dropout, attention_dropout)
        self.contextualize.update_dropout(dropout, attention_dropout)
