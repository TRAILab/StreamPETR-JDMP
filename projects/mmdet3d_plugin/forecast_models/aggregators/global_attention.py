import torch
import torch.nn as nn
from projects.mmdet3d_plugin.forecast_models.aggregators.aggregator import PredictionAggregator
from typing import Dict, Tuple


class GlobalAttention(PredictionAggregator):
    """
    Aggregate context encoding using scaled dot product attention. Query obtained using target agent encoding,
    Keys and values obtained using map and surrounding agent encodings.
    """

    def __init__(self, args: Dict):

        """
        args to include

        enc_size: int Dimension of encodings generated by encoder
        emb_size: int Size of embeddings used for queries, keys and values
        num_heads: int Number of attention heads

        """
        super().__init__()
        self.query_emb = nn.Linear(args['target_agent_enc_size'], args['emb_size'])
        self.key_emb = nn.Linear(args['context_enc_size'], args['emb_size'])
        self.val_emb = nn.Linear(args['context_enc_size'], args['emb_size'])
        self.mha = nn.MultiheadAttention(args['emb_size'], args['num_heads'])

    def forward(self, encodings: Dict) -> torch.Tensor:
        """
        Forward pass for attention aggregator
        """
        target_agent_enc = encodings['target_agent_encoding']
        context_enc = encodings['context_encoding']
        if context_enc['combined'] is not None:
            combined_enc, combined_masks = context_enc['combined'], context_enc['combined_masks'].bool()
        else:
            combined_enc, combined_masks = self.get_combined_encodings(context_enc)

        query = self.query_emb(target_agent_enc).unsqueeze(0)
        keys = self.key_emb(combined_enc).permute(1, 0, 2)
        vals = self.val_emb(combined_enc).permute(1, 0, 2)
        op, _ = self.mha(query, keys, vals, key_padding_mask=combined_masks)
        op = op.squeeze(0)
        op = torch.cat((target_agent_enc, op), dim=-1)

        return op

    @staticmethod
    def get_combined_encodings(context_enc: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates a combined set of map and surrounding agent encodings to be aggregated using attention.
        """
        encodings = []
        masks = []
        if 'map' in context_enc:
            encodings.append(context_enc['map'])
            masks.append(context_enc['map_masks'])
        if 'vehicles' in context_enc:
            encodings.append(context_enc['vehicles'])
            masks.append(context_enc['vehicle_masks'])
        if 'pedestrians' in context_enc:
            encodings.append(context_enc['pedestrians'])
            masks.append(context_enc['pedestrian_masks'])
        combined_enc = torch.cat(encodings, dim=1)
        combined_masks = torch.cat(masks, dim=1).bool()
        return combined_enc, combined_masks
