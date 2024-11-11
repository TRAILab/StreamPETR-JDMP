# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import warnings
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence,
                                         build_attention,
                                         build_feedforward_network)
from mmcv.cnn.bricks.drop import build_dropout
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn import build_norm_layer, xavier_init
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import (ATTENTION,TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.utils import deprecated_api_warning, ConfigDict
import copy
from torch.nn import ModuleList
from .attention import FlashMHA
import torch.utils.checkpoint as cp

from mmcv.runner import auto_fp16
import pickle
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb2d
import math

@ATTENTION.register_module()
class PETRMultiheadFlashAttention(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.
    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=True,
                 **kwargs):
        super(PETRMultiheadFlashAttention, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = True

        self.attn = FlashMHA(embed_dims, num_heads, attn_drop, dtype=torch.float16, device='cuda',
                                          **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.
        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.
        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
        Returns:
            Tensor: forwarded results with shape
            [num_queries, bs, embed_dims]
            if self.batch_first is False, else
            [bs, num_queries embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        out = self.attn(
            q=query,
            k=key,
            v=value,
            key_padding_mask=None)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))


class MultiheadAttentionWrapper(nn.MultiheadAttention):
    def __init__(self, *args, **kwargs):
        super(MultiheadAttentionWrapper, self).__init__(*args, **kwargs)
        self.fp16_enabled = True

    @auto_fp16(out_fp32=True)
    def forward_fp16(self, *args, **kwargs):
        return super(MultiheadAttentionWrapper, self).forward(*args, **kwargs)

    def forward_fp32(self, *args, **kwargs):
        return super(MultiheadAttentionWrapper, self).forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_fp16(*args, **kwargs)
        else:
            return self.forward_fp32( *args, **kwargs)

@ATTENTION.register_module()
class PETRMultiheadAttention(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.
    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 fp16 = False,
                 **kwargs):
        super(PETRMultiheadAttention, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.fp16_enabled = True
        if fp16:
            self.attn = MultiheadAttentionWrapper(embed_dims, num_heads, attn_drop,  **kwargs)
        else:
            self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,  **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.
        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.
        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
        Returns:
            Tensor: forwarded results with shape
            [num_queries, bs, embed_dims]
            if self.batch_first is False, else
            [bs, num_queries embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1).contiguous()
            key = key.transpose(0, 1).contiguous()
            value = value.transpose(0, 1).contiguous()

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]

        if self.batch_first:
            out = out.transpose(0, 1).contiguous()

        return identity + self.dropout_layer(self.proj_drop(out))



@TRANSFORMER_LAYER_SEQUENCE.register_module()
class PETRTransformerEncoder(TransformerLayerSequence):
    """TransformerEncoder of DETR.
    Args:
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`. Only used when `self.pre_norm` is `True`
    """

    def __init__(self, *args, post_norm_cfg=dict(type='LN'), **kwargs):
        super(PETRTransformerEncoder, self).__init__(*args, **kwargs)
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(
                post_norm_cfg, self.embed_dims)[1] if self.pre_norm else None
        else:
            assert not self.pre_norm, f'Use prenorm in ' \
                                      f'{self.__class__.__name__},' \
                                      f'Please specify post_norm_cfg'
            self.post_norm = None

    def forward(self, *args, **kwargs):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        x = super(PETRTransformerEncoder, self).forward(*args, **kwargs)
        if self.post_norm is not None:
            x = self.post_norm(x)
        return x


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class PETRTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 *args,
                 post_norm_cfg=dict(type='LN'),
                 return_intermediate=False,
                 **kwargs):

        super(PETRTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = None

    def forward(self, query, *args, **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        if not self.return_intermediate:
            x = super().forward(query, *args, **kwargs)
            if self.post_norm:
                x = self.post_norm(x)[None]
            return x

        intermediate = []
        for layer in self.layers:
            query = layer(query, *args, **kwargs)
            if self.return_intermediate:
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(query))
                else:
                    intermediate.append(query)
        return torch.stack(intermediate)



@TRANSFORMER.register_module()
class PETRTemporalTransformer(BaseModule):
    """Implements the DETR transformer.
    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:
        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, encoder=None, decoder=None, init_cfg=None, cross=False):
        super(PETRTemporalTransformer, self).__init__(init_cfg=init_cfg)
        if encoder is not None:
            self.encoder = build_transformer_layer_sequence(encoder)
        else:
            self.encoder = None
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.cross = cross

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True


    def forward(self, memory, tgt, query_pos, pos_embed, attn_masks, temp_memory=None, temp_pos=None, mask=None, reg_branch=None):
        """Forward function for `Transformer`.
        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        memory = memory.transpose(0, 1).contiguous()
        query_pos = query_pos.transpose(0, 1).contiguous()
        pos_embed = pos_embed.transpose(0, 1).contiguous()
        
        n, bs, c = memory.shape

        if tgt is None:
            tgt = torch.zeros_like(query_pos)
        else:
            tgt = tgt.transpose(0, 1).contiguous()

        if temp_memory is not None:
            temp_memory = temp_memory.transpose(0, 1).contiguous()
            temp_pos =  temp_pos.transpose(0, 1).contiguous()

        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(
            query=tgt,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_pos,
            temp_memory=temp_memory,
            temp_pos=temp_pos,
            key_padding_mask=mask,
            attn_masks=[attn_masks, None],
            reg_branch=reg_branch,
            )
        out_dec = out_dec.transpose(1, 2).contiguous()
        memory = memory.reshape(-1, bs, c).transpose(0, 1).contiguous()
        return  out_dec, memory


@TRANSFORMER_LAYER.register_module()
class PETRTemporalDecoderLayer(BaseModule):
    """Base `TransformerLayer` for vision transformer.

    It can be built from `mmcv.ConfigDict` and support more flexible
    customization, for example, using any number of `FFN or LN ` and
    use different kinds of `attention` by specifying a list of `ConfigDict`
    named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
    when you specifying `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for `self_attention` or `cross_attention` modules,
            The order of the configs in the list should be consistent with
            corresponding attentions in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for FFN, The order of the configs in the list should be
            consistent with corresponding ffn in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying first element as `norm`.
            Default：None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """

    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=False,
                 with_cp=True,
                 **kwargs):

        deprecated_args = dict(
            feedforward_channels='feedforward_channels',
            ffn_dropout='ffn_drop',
            ffn_num_fcs='num_fcs')
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f'The arguments `{ori_name}` in BaseTransformerLayer '
                    f'has been deprecated, now you should set `{new_name}` '
                    f'and other FFN related arguments '
                    f'to a dict named `ffn_cfgs`. ', DeprecationWarning)
                ffn_cfgs[new_name] = kwargs[ori_name]

        super().__init__(init_cfg)

        self.batch_first = batch_first

        assert set(operation_order) & {
            'self_attn', 'norm', 'ffn', 'cross_attn'} == \
            set(operation_order), f'The operation_order of' \
            f' {self.__class__.__name__} should ' \
            f'contains all four operation type ' \
            f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"

        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn')
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'cross_attn']:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                attention = build_attention(attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            self.ffns.append(
                build_feedforward_network(ffn_cfgs[ffn_index],
                                          dict(type='FFN')))

        self.norms = ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

        self.use_checkpoint = with_cp

    def _forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                temp_memory=None,
                temp_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                if temp_memory is not None:
                    temp_key = temp_value = torch.cat([query, temp_memory], dim=0)
                    temp_pos = torch.cat([query_pos, temp_pos], dim=0)
                else:
                    temp_key = temp_value = query
                    temp_pos = query_pos
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=temp_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query

    def forward(self, 
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                temp_memory=None,
                temp_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs
                ):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if self.use_checkpoint and self.training:
            x = cp.checkpoint(
                self._forward, 
                query,
                key,
                value,
                query_pos,
                key_pos,
                temp_memory,
                temp_pos,
                attn_masks,
                query_key_padding_mask,
                key_padding_mask,
                )
        else:
            x = self._forward(
            query,
            key,
            value,
            query_pos,
            key_pos,
            temp_memory,
            temp_pos,
            attn_masks,
            query_key_padding_mask,
            key_padding_mask,
        )
        return x

@TRANSFORMER.register_module()
class JDMPTemporalTransformer(BaseModule):
    """Implements the DETR transformer.
    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:
        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, encoder=None, decoder=None, init_cfg=None, cross=False):
        super(JDMPTemporalTransformer, self).__init__(init_cfg=init_cfg)
        if encoder is not None:
            self.encoder = build_transformer_layer_sequence(encoder)
        else:
            self.encoder = None
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.cross = cross

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True


    def forward(self, tgt, query_pos, attn_masks, temp_memory=None, temp_pos=None, mask=None, reg_branch=None):
        """Forward function for `Transformer`.
        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - out_dec: Output from decoder. If return_intermediate_dec \
                      is True output has shape [num_dec_layers, bs,
                      num_query, embed_dims], else has shape [1, bs, \
                      num_query, embed_dims].
                - memory: Output results from encoder, with shape \
                      [bs, embed_dims, h, w].
        """
        query_pos = query_pos.transpose(0, 1).contiguous()
        
        if tgt is None:
            tgt = torch.zeros_like(query_pos)
        else:
            tgt = tgt.transpose(0, 1).contiguous()

        if temp_memory is not None:
            temp_memory = temp_memory.transpose(0, 1).contiguous()
            temp_pos =  temp_pos.transpose(0, 1).contiguous()

        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(
            query=tgt,
            key=tgt,
            value=tgt,
            key_pos=query_pos,
            query_pos=query_pos,
            temp_memory=temp_memory,
            temp_pos=temp_pos,
            key_padding_mask=mask,
            attn_masks=attn_masks,
            reg_branch=reg_branch,
            )
        out_dec = out_dec.transpose(1, 2).contiguous()
        return  out_dec

@TRANSFORMER.register_module()
class JDMPForecastTransformer(BaseModule):
    def __init__(self, embed_dims=256, num_propagated=128, num_reg_fcs=2, num_forecast_layers=3, pc_range=None, init_cfg=None):
        super(JDMPForecastTransformer, self).__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.num_propagated = num_propagated
        self.num_reg_fcs = num_reg_fcs
        self.num_forecast_layers = num_forecast_layers
        self.pc_range = nn.Parameter(torch.tensor(pc_range), requires_grad=False)

        anchor_infos = pickle.load(open('ckpts/motion_anchor_infos_mode6.pkl', 'rb'))
        self.kmeans_anchors = torch.stack(
            [torch.from_numpy(a) for a in anchor_infos["anchors_all"]]).float()
        self.kmeans_anchors = self.kmeans_anchors[:3].reshape(-1, 12, 2)
        self.num_forecast_modes = self.kmeans_anchors.size(0)

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""

        traj_cls_branch = []
        for _ in range(self.num_reg_fcs):
            traj_cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            traj_cls_branch.append(nn.LayerNorm(self.embed_dims))
            traj_cls_branch.append(nn.ReLU(inplace=True))
        traj_cls_branch.append(nn.Linear(self.embed_dims, 1))
        traj_cls_branch = nn.Sequential(*traj_cls_branch)
        self.traj_cls_branches = nn.ModuleList([copy.deepcopy(traj_cls_branch) for i in range(self.num_forecast_layers)])

        traj_reg_branch = []
        for _ in range(self.num_reg_fcs):
            traj_reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            traj_reg_branch.append(nn.ReLU())
        traj_reg_branch.append(nn.Linear(self.embed_dims, 12 * 2))
        traj_reg_branch = nn.Sequential(*traj_reg_branch)
        self.traj_reg_branches = nn.ModuleList([copy.deepcopy(traj_reg_branch) for i in range(self.num_forecast_layers)])

        self.learnable_motion_query_embedding = nn.Embedding(
            self.num_forecast_modes, self.embed_dims)

        self.forecast_det_query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.forecast_agent_level_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.forecast_scene_level_ego_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.forecast_scene_level_offset_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.intention_interaction_layers = nn.TransformerEncoderLayer(d_model=256,
                                                                nhead=8,
                                                                dropout=0.1,
                                                                dim_feedforward=256*2,
                                                                batch_first=True)
        self.static_dynamic_fuser = nn.Sequential(
            nn.Linear(self.embed_dims*2, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )
        self.dynamic_embed_fuser = nn.Sequential(
            nn.Linear(self.embed_dims*3, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )
        self.in_query_fuser = nn.Sequential(
            nn.Linear(self.embed_dims*2, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )
        self.out_query_fuser = nn.Sequential(
            nn.Linear(self.embed_dims*2, self.embed_dims*2),
            nn.ReLU(),
            nn.Linear(self.embed_dims*2, self.embed_dims),
        )
        self.detection_agent_interaction_layers = nn.ModuleList(
            [nn.TransformerDecoderLayer(d_model=256,
                                        nhead=8,
                                        dropout=0.1,
                                        dim_feedforward=256*2,
                                        batch_first=True) 
            for i in range(self.num_forecast_layers)])
        
        self.unflatten_traj = nn.Unflatten(3, (12, 2))
        self.log_softmax = nn.LogSoftmax(dim=2)

    def init_weights(self):
        # follow the official DETR to init parameters
        # for m in self.modules():
        #     if hasattr(m, 'weight') and m.weight.dim() > 1:
        #         xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, detection_query, detection_reference_pose):
        B = detection_query.size(0)
        A = self.num_propagated
        M = self.num_forecast_modes
        T = 12
        
        # Detection query
        detection_reference_point = detection_reference_pose[..., :2]
        detection_reference_point_norm = self.norm_points_2d(detection_reference_point)
        detection_query_pos = self.forecast_det_query_embedding(pos2posemb2d(detection_reference_point_norm))

        agent_level_anchors = self.kmeans_anchors.to(detection_reference_point.device).detach() # M, 12, 2
        scene_level_ego_anchors = self.anchor_coordinate_transform(agent_level_anchors, detection_reference_pose) # B, A, M, 12, 2
        scene_level_offset_anchors = self.anchor_coordinate_transform(agent_level_anchors, detection_reference_pose, with_translation_transform=False)  
        
        agent_level_norm = self.norm_points_2d(agent_level_anchors) # M, 12, 2
        scene_level_ego_norm = self.norm_points_2d(scene_level_ego_anchors) # B, A, M, 12, 2
        scene_level_offset_norm = self.norm_points_2d(scene_level_offset_anchors)

        agent_level_embedding = self.forecast_agent_level_embedding(pos2posemb2d(agent_level_norm[..., -1, :]))  # M, C
        scene_level_ego_embedding = self.forecast_scene_level_ego_embedding(pos2posemb2d(scene_level_ego_norm[..., -1, :])) # B, A, M, C
        scene_level_offset_embedding = self.forecast_scene_level_offset_embedding(pos2posemb2d(scene_level_offset_norm[..., -1, :])) 

        agent_level_embedding = agent_level_embedding[None,None, ...].expand(B, A, -1, -1) # B, A, M, C
        learnable_query_pos = self.learnable_motion_query_embedding.weight.to(detection_query.device) # M, C
        learnable_embed = learnable_query_pos[None, None, ...].expand(B, A, -1, -1) # B, A, M, C

        init_reference  = scene_level_offset_anchors

        detection_query_bc = detection_query.unsqueeze(2).expand(-1, -1, M, -1)  # B, A, M, C
        detection_query_pos_bc = detection_query_pos.unsqueeze(2).expand(-1, -1, M, -1)  # B, A, M, C

        agent_level_embedding = torch.flatten(agent_level_embedding, start_dim=0, end_dim=1) # B*A, M, C
        agent_level_embedding = self.intention_interaction_layers(agent_level_embedding).view(B, A, M, -1)

        static_intention_embed = agent_level_embedding + scene_level_offset_embedding + learnable_embed
        reference_trajs_input = init_reference.unsqueeze(4).detach()

        query_embed = torch.zeros_like(static_intention_embed)

        intermediate = []
        intermediate_reference_trajs = []

        for lid in range(self.num_forecast_layers):

            dynamic_query_embed = self.dynamic_embed_fuser(torch.cat(
                [agent_level_embedding, scene_level_offset_embedding, scene_level_ego_embedding], dim=-1))
            
            query_embed_intention = self.static_dynamic_fuser(torch.cat(
                [static_intention_embed, dynamic_query_embed], dim=-1))  # B, A, M, C
            
            query_embed = self.in_query_fuser(torch.cat([query_embed, query_embed_intention], dim=-1))
            
            q_dai = query_embed + detection_query_pos_bc
            q_dai = torch.flatten(q_dai, start_dim=0, end_dim=1) # B*A, M, C
            k_dai = (detection_query + detection_query_pos).reshape(B*A, -1).unsqueeze(1).expand(B*A, M, -1) # B, A, C -> B*A, M, C
            detection_query_embed = self.detection_agent_interaction_layers[lid](q_dai, k_dai).view(B, A, M, -1)
            
            query_embed = [detection_query_embed, detection_query_bc+detection_query_pos_bc]
            query_embed = torch.cat(query_embed, dim=-1)
            query_embed = self.out_query_fuser(query_embed)

            tmp = self.traj_reg_branches[lid](query_embed)
            tmp = tmp.view(B, A, M, T, -1)

            tmp[..., :2] = torch.cumsum(tmp[..., :2], dim=3)
            new_reference_trajs = torch.zeros_like(reference_trajs_input)
            new_reference_trajs = tmp[..., :2]
            reference_trajs = new_reference_trajs.detach()
            reference_trajs_input = reference_trajs.unsqueeze(4)  # B A N 12 NUM_LEVEL  2

            ep_offset_embed = reference_trajs.detach()
            ep_ego_embed = self.trajectory_coordinate_transform(reference_trajs, detection_reference_pose, with_rotation_transform=False).detach()
            ep_agent_embed = self.trajectory_coordinate_transform(reference_trajs, detection_reference_pose, with_translation_transform=False).detach()

            agent_level_embedding = self.forecast_agent_level_embedding(pos2posemb2d(self.norm_points_2d(ep_agent_embed[..., -1, :])))
            scene_level_ego_embedding = self.forecast_scene_level_ego_embedding(pos2posemb2d(self.norm_points_2d(ep_ego_embed[..., -1, :])))
            scene_level_offset_embedding = self.forecast_scene_level_offset_embedding(pos2posemb2d(self.norm_points_2d(ep_offset_embed[..., -1, :])))

            intermediate.append(query_embed)
            intermediate_reference_trajs.append(reference_trajs)

        inter_states = torch.stack(intermediate)

        outputs_traj_scores = []
        outputs_trajs = []

        for lvl in range(inter_states.shape[0]):
            outputs_class = self.traj_cls_branches[lvl](inter_states[lvl])
            tmp = self.traj_reg_branches[lvl](inter_states[lvl])
            tmp = self.unflatten_traj(tmp)
            
            # we use cumsum trick here to get the trajectory 
            tmp[..., :2] = torch.cumsum(tmp[..., :2], dim=3)

            # outputs_class = self.log_softmax(outputs_class.squeeze(3))
            outputs_traj_scores.append(outputs_class)

            # for bs in range(tmp.shape[0]):
            #     tmp[bs] = self.bivariate_gaussian_activation(tmp[bs])
            outputs_trajs.append(tmp)
        outputs_traj_scores = torch.stack(outputs_traj_scores)
        outputs_trajs = torch.stack(outputs_trajs)

        return outputs_trajs, outputs_traj_scores

    def anchor_coordinate_transform(self, anchors, ref_poses, with_translation_transform=True, with_rotation_transform=True):
        """
        Transform anchor coordinates with respect to detected bounding boxes in the batch.

        Args:
            anchors (torch.Tensor): A tensor containing the k-means anchor values.
            ref_poses (torch.Tensor): A tensor of the detection position and rotation each sample in the batch.
            with_translate (bool, optional): Whether to perform translation transformation. Defaults to True.
            with_rot (bool, optional): Whether to perform rotation transformation. Defaults to True.

        Returns:
            torch.Tensor: A tensor containing the transformed anchor coordinates.
        """
        ref_poses_vec = ref_poses.clone()
        # use_velo_for_rot = False
        # if use_velo_for_rot:
        #     velo_mag_thresh = 1.0
        #     velo = self.memory_velo.clone()
        #     velo_dir = torch.atan2(velo[..., 1], velo[..., 0])
        #     velo_mag = torch.norm(velo[..., :2], dim=-1)
        #     ref_poses_vec[..., 2] = torch.where(velo_mag > velo_mag_thresh, velo_dir, ref_poses_vec[..., 2])
        ref_poses_vec[..., 2] -= torch.tensor(math.pi)/2 
        ref_poses_mat = self.pose2d_vec_to_mat(ref_poses_vec).unsqueeze(2).unsqueeze(2) # B, A, M, T, 3, 3
        if not with_translation_transform:
            ref_poses_mat[..., :2, 2] = 0.0
        if not with_rotation_transform:
            ref_poses_mat[..., :2, :2] = torch.eye(2)
        transformed_anchors = anchors[None, None, ...] # B, A, M, T, 2
        transformed_anchors = torch.cat([transformed_anchors, torch.ones_like(transformed_anchors[..., :1])], dim=-1) # B, A, M, T, 3
        transformed_anchors = torch.matmul(ref_poses_mat, transformed_anchors.unsqueeze(-1)).squeeze(-1) # B, A, M, T, 3
        transformed_anchors = transformed_anchors[..., :2]
        # viz
        # import matplotlib.pyplot as plt
        # import numpy as np
        # output_path = 'output/uniad_anchors/transformed_anchor_viz.png'
        # for i in range(transformed_anchors.shape[0]):
        #     fig1, ax1 = plt.subplots()
        #     b_id, a_id = i, i 
        #     for j in range(transformed_anchors.shape[2]):
        #         points = transformed_anchors[b_id, a_id, j].detach().cpu().numpy()
        #         ax1.plot(points[:, 0], points[:, 1], 'o-')
        #     velo = self.memory_velo[b_id, a_id].detach().cpu().numpy()
        #     angle = ref_poses[b_id, a_id, 2].detach().cpu().numpy()
        #     pos = ref_poses[b_id, a_id, :2].detach().cpu().numpy()
        #     ax1.plot([pos[0], pos[0] + 80*np.cos(angle)], [pos[1], pos[1] + 80*np.sin(angle)], '*-', color='r', alpha=0.5, label='ang_pred_80m')
        #     ax1.plot([pos[0], pos[0] + 6*velo[0]], [pos[1], pos[1] + 6*velo[1]], '*-', color='k', alpha=0.5, label='velo_pred')
        #     ax1.legend()
        #     fig1.savefig(output_path)
        #     plt.close(fig1)
        #     print(f"Plot saved to {output_path}")
        #     breakpoint()
        return transformed_anchors


    def trajectory_coordinate_transform(self, trajectory, ref_poses, with_translation_transform=True, with_rotation_transform=True):
        """
        Transform trajectory coordinates with respect to detected bounding boxes in the batch.
        Args:
            trajectory (torch.Tensor): predicted trajectory.
            ref_poses (torch.Tensor): A tensor of the detection position and rotation each sample in the batch.
            with_translate (bool, optional): Whether to perform translation transformation. Defaults to True.
            with_rot (bool, optional): Whether to perform rotation transformation. Defaults to True.

        Returns:
            torch.Tensor: A tensor containing the transformed trajectory coordinates.
        """
        ref_poses_vec = ref_poses.clone()
        ref_poses_vec[..., 2] -= torch.tensor(math.pi)/2 
        ref_poses_vec[..., 2] = -ref_poses_vec[..., 2]
        ref_poses_mat = self.pose2d_vec_to_mat(ref_poses_vec).unsqueeze(2).unsqueeze(2) # B, A, M, T, 3, 3
        if not with_translation_transform:
            ref_poses_mat[..., :2, 2] = 0.0
        if not with_rotation_transform:
            ref_poses_mat[..., :2, :2] = torch.eye(2)
        trajectory = torch.cat([trajectory, torch.ones_like(trajectory[..., :1])], dim=-1) # B, A, M, T, 3
        trajectory = torch.matmul(ref_poses_mat, trajectory.unsqueeze(-1)).squeeze(-1) # B, A, M, T, 2
        trajectory = trajectory[..., :2]
        # viz
        # import matplotlib.pyplot as plt
        # import numpy as np
        # output_path = 'output/uniad_anchors/transformed_anchor_viz.png'
        # for i in range(trajectory.shape[0]):
        #     fig1, ax1 = plt.subplots()
        #     b_id, a_id = i, i 
        #     for j in range(trajectory.shape[2]):
        #         points = trajectory[b_id, a_id, j].detach().cpu().numpy()
        #         ax1.plot(points[:, 0], points[:, 1], 'o-')
        #     velo = self.memory_velo[b_id, a_id].detach().cpu().numpy()
        #     angle = ref_poses[b_id, a_id, 2].detach().cpu().numpy()
        #     pos = ref_poses[b_id, a_id, :2].detach().cpu().numpy()
        #     ax1.plot([pos[0], pos[0] + 80*np.cos(angle)], [pos[1], pos[1] + 80*np.sin(angle)], '*-', color='r', alpha=0.5, label='ang_pred_80m')
        #     ax1.plot([pos[0], pos[0] + 6*velo[0]], [pos[1], pos[1] + 6*velo[1]], '*-', color='k', alpha=0.5, label='velo_pred')
        #     ax1.legend()
        #     fig1.savefig(output_path)
        #     plt.close(fig1)
        #     print(f"Plot saved to {output_path}")
        #     breakpoint()

        return trajectory
    
    def norm_points_2d(self, points):
        points = (points - self.pc_range[:2]) / (self.pc_range[3:5] - self.pc_range[0:2])
        return points
    
    def bivariate_gaussian_activation(self, ip):
        """
        Activation function to output parameters of bivariate Gaussian distribution.

        Args:
            ip (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor containing the parameters of the bivariate Gaussian distribution.
        """
        mu_x = ip[..., 0:1]
        mu_y = ip[..., 1:2]
        sig_x = ip[..., 2:3]
        sig_y = ip[..., 3:4]
        rho = ip[..., 4:5]
        sig_x = torch.exp(sig_x)
        sig_y = torch.exp(sig_y)
        rho = torch.tanh(rho)
        out = torch.cat([mu_x, mu_y, sig_x, sig_y, rho], dim=-1)
        return out

    def pose2d_vec_to_mat(self, pose2d_vec):
        pose2d_mat = torch.eye(3, device=pose2d_vec.device).expand(*pose2d_vec.size()[:-1], 3, 3).clone()
        pose2d_mat[..., 0, 0] = torch.cos(pose2d_vec[..., 2])
        pose2d_mat[..., 0, 1] = -torch.sin(pose2d_vec[..., 2])
        pose2d_mat[..., 1, 0] = torch.sin(pose2d_vec[..., 2])
        pose2d_mat[..., 1, 1] = torch.cos(pose2d_vec[..., 2])
        pose2d_mat[..., 0:2, 2] = pose2d_vec[..., 0:2]
        return pose2d_mat
