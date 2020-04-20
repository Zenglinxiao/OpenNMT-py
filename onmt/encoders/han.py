"""
 Implementation of "Document-Level Neural Machine Translation
 with Hierarchical Attention Networks"
"""

from onmt.encoders.transformer import TransformerEncoder
from onmt.modules.han_context import HierarquicalContext


class HANEncoder(TransformerEncoder):
    """The HAN encoder from "Document-Level Neural Machine Translation
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

    def __init__(self, num_layers, d_model, heads, d_ff, dropout,
                 attention_dropout, embeddings, max_relative_positions,
                 context_size, mode, padding_idx, bos_idx=None, eos_idx=None):
        super(HANEncoder, self).__init__(
            num_layers, d_model, heads, d_ff, dropout,
            attention_dropout, embeddings, max_relative_positions)
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
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions,
            context_size=opt.n_src_ctxs,
            mode=opt.encoder_mode,
            padding_idx=embeddings.word_padding_idx,
            bos_idx=embeddings.word_bos_idx,
            eos_idx=embeddings.word_eos_idx)

    def forward(self, src, lengths=None, ctxs=None, **kwargs):
        """See :func:`EncoderBase.forward()`"""
        # 1. Transformer part forward
        emb, out, lengths = super().forward(src, lengths)
        # 2. Context part
        if self.mode in ['all', 'context']:
            emb_ctxs, out_ctxs, lengths_ctxs = self.ctxs_forward(ctxs)
            out, attn_word_enc_l, attn_sent_enc = self.contextualize(
                out, ctxs, out_ctxs)
        return emb, out, lengths

    def ctxs_forward(self, ctxs):
        """Base TransformerEncoder forward for all ctxs."""
        emb_ctxs, out_ctxs, lengths_ctxs = [], [], []
        if ctxs is not None:
            for ctx in ctxs:
                ctx, ctx_lengths = ctx if isinstance(ctx, tuple) \
                    else (ctx, None)
                emb_ctx, out_ctx, lengths_ctx = super().forward(
                    ctx, ctx_lengths)
                emb_ctxs.append(emb_ctx)
                out_ctxs.append(out_ctx)
                lengths_ctxs.append(lengths_ctx)
        return emb_ctxs, out_ctxs, lengths_ctxs

    def update_dropout(self, dropout, attention_dropout):
        super().update_dropout(dropout, attention_dropout)
        self.contextualize.update_dropout(dropout, attention_dropout)
