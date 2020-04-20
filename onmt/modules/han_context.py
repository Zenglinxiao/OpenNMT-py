""" Hierarquical Context module """
import torch
import torch.nn as nn
from onmt.modules import MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward


class HierarquicalContext(nn.Module):
    """
    Hierarquical Context module of HAN.

    Args:
      d_model: hidden layers size
      dropout: dropout for each layer
      d_ff: Position-wise Feed-Forward hidden layers size
      heads: number of heads for the multi-head attention
      context_size: number of previous sentences
      padding_idx: id for padding word
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 context_size, padding_idx, bos_idx=None, eos_idx=None):
        super().__init__()
        self.context_size = context_size
        self.padding_idx = padding_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

        self.layer_norm_query_word = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_query_sent = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_word = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_sent = nn.LayerNorm(d_model, eps=1e-6)

        self.word_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout)
        self.sent_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.linear = nn.Linear(2*d_model, d_model)
        self.sigmoid = nn.Sigmoid()

    def forward(self, query, ctxs, contexts):
        """HierarquicalContext module forward.

        Args:
            query (FloatTensor): Enc/Dec hidden states of current sent.
                    {sent_len, batch, dim}
            ctxs (list[tuple[LongTensor]]): previous sentences w.r.t inputs.
                    used to extract masks for contexts.
                    [{context_len, batch, n_feat}, {batch,}]
            contexts (list[LongTensor]): previous sentence's hidden.
                    [{context_len, batch, dim}]
            NOTE: context_len could be different across No.context.

        Returns:
            (FloatTensor, list[FloatTensor], FloatTensor)
            * out : Final current contextualized Enc/Dec hidden states.
                    {sent_len, batch, dim}
            * attn_word_l : list of Word-level attention weights.
                     [{batch, heads, sent_len, context_len}]
            * attn_sent : Sentence-level attention weights
                     {batch, sent_len, heads, 1, n_context}
        """
        query = query.transpose(0, 1).contiguous()  # (B, Lq, D)
        query_word_norm = self.layer_norm_query_word(query)  # NOTE: neccessary?
        context_word_l, attn_word_l = [], []
        ctx_sent_mask_l = []
        for ctx, context in zip(ctxs, contexts):
            ctx, ctx_length = ctx if isinstance(ctx, tuple) else (ctx, None)
            ctx_ = ctx.transpose(0, 1).contiguous()
            ctx_word_mask_, ctx_sent_mask_ = self.get_masks(ctx_)
            ctx_sent_mask_l.append(ctx_sent_mask_)
            context_ = context.transpose(0, 1).contiguous()
            context_word_, attn_word_ = self.word_level_attn(
                query_word_norm, context_, ctx_word_mask_)
            context_word_l.append(context_word_)  # [(B, Lq, D)]
            attn_word_l.append(attn_word_)  # [(B, H, Lq, *Lc)]

        # context_word_l, attn_word_l = zip(
        #     self.word_level_attn(query, context) for context in contexts)
        context_word = torch.stack(context_word_l, dim=1)  # (B, n_ctx, Lq, D)

        # Sentence-level
        # b_size, n_ctx, len_q, dim = context_word.size()
        # context_sent = context_word.view(b_size*n_ctx, -1, dim)  # NOTE: ?? useless
        # context_sent = self.layer_norm_word(context_sent)  # TODO: maybe move forward
        # context_sent = context_sent.view(b_size, n_ctx, len_q, dim)
        # context_sent = context_sent.transpose(1, 2).contiguous()  # (B, Lq, nc, D)
        # context_sent = context_sent.view(b_size*len_q, n_ctx, dim)  # (B*Lq, nc, D)

        # query_sent_norm = self.layer_norm_query_sent(query)
        # b_size_, len_q_, dim_ = query_sent_norm.size()
        # query_sent_norm = query_sent_norm.view(b_size_*len_q_, 1, dim_)  # (B*Lq, 1, D)

        # ctx_sent_mask = torch.stack(ctx_sent_mask_l, dim=1)  # (B, n_ctx)
        # context_sent, attn_sent = self.sent_attn(
        #     context_sent, context_sent, query_sent_norm,
        #     mask=ctx_sent_mask, attn_type="context")

        ctx_sent_mask = torch.stack(ctx_sent_mask_l, dim=1)  # (B, n_ctx)
        context_sent, attn_sent = self.sent_level_attn(
            query, context_word, ctx_sent_mask)
        out = self.context_gating(query, context_sent)
        return out.transpose(0, 1).contiguous(), attn_word_l, attn_sent

    def get_masks(self, ctx):
        """Return word & sentence mask for a given ctx.

        Args:
            ctx (LongTensor): Enc/Dec previous inputs.(B, Lc, n_feat)
        Returns:
            (BoolTensor, BoolTensor)
            * ctx_word_mask : (B, Lc)
            * ctx_sent_mask : (B,)
        """
        ctx_base = ctx[:, :, 0]
        ctx_pad_mask = ctx_base.eq(self.padding_idx)  # (B, Lc)
        ctx_special_mask = ctx_pad_mask  # (B, Lc)
        if self.bos_idx is not None:
            ctx_special_mask |= ctx_base.eq(self.bos_idx)
        if self.eos_idx is not None:
            ctx_special_mask |= ctx_base.eq(self.eos_idx)
        # create a sentence mask from special_token_mask
        # True for sentence that should be masked
        ctx_sent_mask = ctx_special_mask.all(dim=1)  # (B,)
        ctx_word_mask = ctx_pad_mask
        if self.bos_idx is not None or self.eos_idx is not None:
            ctx_word_mask |= ctx_sent_mask.unsqueeze(1).expand(
                -1, ctx_pad_mask.size(1))
        return ctx_word_mask, ctx_sent_mask

    def word_level_attn(self, query, context, ctx_word_mask):
        """Do word level attn on query (src/tgt) and context.

        Args:
            query (FloatTensor): hidden states of current sent. (B, Lq, D)
            context (FloatTensor): hidden states of previous sent. (B, Lc, D)
            ctx_word_mask (BoolTensor): (B, Lc)

        Returns:
            (FloatTensor, FloatTensor)
            * context_word (B, Lq, D): word-wise contextualized sentence.
            * attn_word (B, H, Lq, Lc): multi-head attention query vs. context.
        """
        # CHECKS
        # b_size_q, len_q, dim_q = query.size()
        # b_size_c, len_c, dim_c = context.size()
        # aeq(b_size_q, b_size_c)
        # aeq(dim_q, dim_c)
        # CHECKS ENDS
        ctx_word_mask_ = ctx_word_mask.unsqueeze(1)  # (B, 1, Lc)
        # NOTE: layernorm context? : query is LNed but not context !
        context_word, attn_word = self.word_attn(
            context, context, query, mask=ctx_word_mask_, attn_type="context")
        return context_word, attn_word

    def sent_level_attn(self, query, context_word, ctx_sent_mask):
        """Do sentence level attn on query (src/tgt) and contextualized vector.

        Args:
            query (FloatTensor): hidden states of current sent. (B, Lq, D)
            context_word (FloatTensor): contextual vector. (B, n_ctx, Lq, D)
            ctx_sent_mask (BoolTensor): sentence mask. (B, n_ctx)

        Returns:
            (FloatTensor, FloatTensor)
            * context_sent (B, Lq, D): contextualized word.
            * attn_sent (B, Lq, H, 1, n_ctx): multi-head attention.
        """
        # b_size, n_ctx, len_q, dim = context_word.size()
        # query_sent = query.unsqueeze(1).expand(-1, n_ctx, -1, -1)
        # context_sent, attn_sent = self.sent_attn(
        #     context_word, context_word, query_sent,
        #     mask=ctx_sent_mask, attn_type="context")
        b_size, n_ctx, len_q, dim = context_word.size()
        context_word_ = self.layer_norm_word(context_word)  # NOTE
        context_word_ = context_word_.transpose(1, 2).contiguous()  # (B, Lq, nc, D)
        context_word_ = context_word_.view(b_size*len_q, n_ctx, dim)  # (B*Lq, nc, D)

        query_sent_norm = self.layer_norm_query_sent(query)  # NOTE: neccessary ?
        b_size_, len_q_, dim_ = query_sent_norm.size()
        # aeq(b_size_, b_size)
        # aeq(len_q, len_q_)
        query_sent_norm = query_sent_norm.view(b_size_*len_q_, 1, dim_)  # (B*Lq, 1, D)

        ctx_sent_mask_ = ctx_sent_mask.unsqueeze(1)\
                                      .expand(b_size, len_q, n_ctx)\
                                      .contiguous().view(b_size*len_q, -1)\
                                      .unsqueeze(1).contiguous()  # (B*Lq, 1, n_ctx)

        context_sent, attn_sent = self.sent_attn(
            context_word_, context_word_, query_sent_norm,
            mask=ctx_sent_mask_, attn_type="context")
        context_sent = context_sent.view(b_size, len_q, dim)
        attn_sent = attn_sent.view(b_size, len_q, -1, 1, n_ctx)
        context_sent = self.feed_forward(context_sent)
        return context_sent, attn_sent

    def context_gating(self, query, context_sent):
        """Regulate sentence-level & doc-level info by a gate.

        Args:
            query (FloatTensor): hidden states of current sent. (B, Lq, D)
            context_sent (FloatTensor): contextualized word. (B, Lq, D)

        Returns:
            out (FloatTensor): final hidden state for a word. (B, Lq, D)
        """
        sent_doc = torch.cat([query, context_sent], dim=2)  # (B, Lq, 2D)
        lambda_t = self.sigmoid(self.linear(sent_doc))  # (B, Lq, D)
        out = (1 - lambda_t) * query + lambda_t * context_sent

        out = self.layer_norm_sent(out)
        return out

    def update_dropout(self, dropout, attention_dropout):
        self.word_attn.update_dropout(attention_dropout)
        self.sent_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
