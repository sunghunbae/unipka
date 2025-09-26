import torch
import torch.nn as nn
import argparse
import numpy as np
import logging

from .dictionary import Dictionary, DICT, DICT_CHARGE
from .layers import (
    TransformerEncoderWithPair, 
    NonLinearHead, 
    GaussianLayer, 
    ClassificationHead,
    pad_1d_tokens,
    pad_2d,
    pad_coords,
)


BACKBONE = {'transformer': TransformerEncoderWithPair}
logger = logging.getLogger(__name__)


def molecule_architecture():
    args = argparse.ArgumentParser()
    args.encoder_layers = getattr(args, "encoder_layers", 15)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.backbone = getattr(args, "backbone", "transformer")
    args.kernel = getattr(args, "kernel", "gaussian")
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)
    return args



class UniMolModel(nn.Module):
    """
    UniMolModel is a specialized model for molecular, protein, crystal, or MOF (Metal-Organic Frameworks) data. 
    It dynamically configures its architecture based on the type of data it is intended to work with. The model
    supports multiple data types and incorporates various architecture configurations and pretrained weights.

    Attributes:
        - output_dim: The dimension of the output layer.
        - data_type: The type of data the model is designed to handle.
        - remove_hs: Flag to indicate whether hydrogen atoms are removed in molecular data.
        - pretrain_path: Path to the pretrained model weights.
        - dictionary: The dictionary object used for tokenization and encoding.
        - mask_idx: Index of the mask token in the dictionary.
        - padding_idx: Index of the padding token in the dictionary.
        - embed_tokens: Embedding layer for token embeddings.
        - encoder: Transformer encoder backbone of the model.
        - gbf_proj, gbf: Layers for Gaussian basis functions or numerical embeddings.
        - classification_head: The final classification head of the model.
    """
    def __init__(self, model_path, output_dim=2, **params):
        """
        Initializes the UniMolModel with specified parameters and data type.

        :param output_dim: (int) The number of output dimensions (classes).
        :param data_type: (str) The type of data (e.g., 'molecule', 'protein').
        :param params: Additional parameters for model configuration.
        """
        super().__init__()
        self.args = molecule_architecture()
        self.output_dim = output_dim
        self.remove_hs = params.get('remove_hs', False)
        self.pretrain_path = model_path
        self.head_name = params.get('head_name', 'chembl_all')
        self.dict_dir = params.get('dict_dir', 'dict')
        self.dictionary = Dictionary.load_from_str(DICT)
        self.mask_idx = self.dictionary.add_symbol("[MASK]", is_special=True)
        self.padding_idx = self.dictionary.pad()
        self.embed_tokens = nn.Embedding(
            len(self.dictionary), self.args.encoder_embed_dim, self.padding_idx
        )
        self.charge_dictionary = Dictionary.load_from_str(DICT_CHARGE)
        self.charge_mask_idx = self.charge_dictionary.add_symbol("[MASK]", is_special=True)
        self.charge_padding_idx = self.charge_dictionary.pad()
        self.embed_charges = nn.Embedding(
            len(self.charge_dictionary), self.args.encoder_embed_dim, self.charge_padding_idx
        )
        self.encoder = BACKBONE[self.args.backbone](
            encoder_layers=self.args.encoder_layers,
            embed_dim=self.args.encoder_embed_dim,
            ffn_embed_dim=self.args.encoder_ffn_embed_dim,
            attention_heads=self.args.encoder_attention_heads,
            emb_dropout=self.args.emb_dropout,
            dropout=self.args.dropout,
            attention_dropout=self.args.attention_dropout,
            activation_dropout=self.args.activation_dropout,
            max_seq_len=self.args.max_seq_len,
            activation_fn=self.args.activation_fn,
            no_final_head_layer_norm=self.args.delta_pair_repr_norm_loss < 0,
        )
        K = 128
        n_edge_type = len(self.dictionary) * len(self.dictionary)
        self.gbf_proj = NonLinearHead(
            K, self.args.encoder_attention_heads, self.args.activation_fn
        )
        if self.args.kernel == 'gaussian':
            self.gbf = GaussianLayer(K, n_edge_type)
        self.classification_heads = nn.ModuleDict()
        self.classification_heads[self.head_name] = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=self.args.encoder_embed_dim,
            num_classes=self.output_dim,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )
        self.load_pretrained_weights(path=self.pretrain_path)

    def load_pretrained_weights(self, path):
        """
        Loads pretrained weights into the model.

        :param path: (str) Path to the pretrained weight file.
        """
        if path is not None:
            logger.info("Loading pretrained weights from {}".format(path))
            state_dict = torch.load(path, weights_only=False, map_location=lambda storage, loc: storage)
            errors = self.load_state_dict(state_dict['model'], strict=True)
            if errors.missing_keys:
                logger.warning(
                    "Error in loading model state, missing_keys "
                    + str(errors.missing_keys)
                )
            if errors.unexpected_keys:
                logger.warning(
                    "Error in loading model state, unexpected_keys "
                    + str(errors.unexpected_keys)
                )

    @classmethod
    def build_model(cls, args):
        """
        Class method to build a new instance of the UniMolModel.

        :param args: Arguments for model configuration.
        :return: An instance of UniMolModel.
        """
        return cls(args)

    def forward(
        self,
        src_tokens,
        src_charges,
        src_distance,
        src_coord,
        src_edge_type,
        return_repr=False,
        return_atomic_reprs=False,
        **kwargs
    ):
        """
        Defines the forward pass of the model.

        :param src_tokens: Tokenized input data.
        :param src_distance: Additional molecular features.
        :param src_coord: Additional molecular features.
        :param src_edge_type: Additional molecular features.
        :param gas_id: Optional environmental features for MOFs.
        :param gas_attr: Optional environmental features for MOFs.
        :param pressure: Optional environmental features for MOFs.
        :param temperature: Optional environmental features for MOFs.
        :param return_repr: Flags to return intermediate representations.
        :param return_atomic_reprs: Flags to return intermediate representations.

        :return: Output logits or requested intermediate representations.
        """
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None
        x = self.embed_tokens(src_tokens)
        # involve charge info
        charge_padding_mask = src_charges.eq(self.charge_padding_idx)
        if not charge_padding_mask.any():
            padding_mask = None
        charges_emb = self.embed_charges(src_charges)
        x += charges_emb

        def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = self.gbf(dist, et)
            gbf_result = self.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias
        graph_attn_bias = get_dist_features(src_distance, src_edge_type)
        (
            encoder_rep,
            _,
            _,
            _,
            _,
        ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
        cls_repr = encoder_rep[:, 0, :]  # CLS token repr
        all_repr = encoder_rep[:, :, :]  # all token repr

        filtered_tensors = []
        filtered_coords = []
        for tokens, coord in zip(src_tokens, src_coord):
            filtered_tensor = tokens[(tokens != 0) & (tokens != 1) & (tokens != 2)] # filter out BOS(0), EOS(1), PAD(2)
            filtered_coord = coord[(tokens != 0) & (tokens != 1) & (tokens != 2)]
            filtered_tensors.append(filtered_tensor)
            filtered_coords.append(filtered_coord)

        lengths = [len(filtered_tensor) for filtered_tensor in filtered_tensors] # Compute the lengths of the filtered tensors
        if return_repr and return_atomic_reprs:
            cls_atomic_reprs = [] 
            atomic_symbols = []
            for i in range(len(all_repr)):
                atomic_reprs = encoder_rep[i, 1:lengths[i]+1, :]
                atomic_symbol = []
                for atomic_num in filtered_tensors[i]:
                    atomic_symbol.append(self.dictionary.symbols[atomic_num])
                atomic_symbols.append(atomic_symbol)
                cls_atomic_reprs.append(atomic_reprs)
            return {
                'cls_repr': cls_repr, 
                'atomic_symbol': atomic_symbols, 
                'atomic_coords': filtered_coords, 
                'atomic_reprs': cls_atomic_reprs
                }        
        if return_repr and not return_atomic_reprs:
            return {'cls_repr': cls_repr}  

        logits = self.classification_heads[self.head_name](cls_repr)
        return logits

    def batch_collate_fn(self, samples):
        """
        Custom collate function for batch processing non-MOF data.

        :param samples: A list of sample data.

        :return: A tuple containing a batch dictionary and labels.
        """
        batch = {}
        for k in samples[0][0].keys():
            if k == 'src_coord':
                v = pad_coords([torch.tensor(s[0][k]).float() for s in samples], pad_idx=0.0)
            elif k == 'src_edge_type':
                v = pad_2d([torch.tensor(s[0][k]).long() for s in samples], pad_idx=self.padding_idx)
            elif k == 'src_distance':
                v = pad_2d([torch.tensor(s[0][k]).float() for s in samples], pad_idx=0.0)
            elif k == 'src_tokens':
                v = pad_1d_tokens([torch.tensor(s[0][k]).long() for s in samples], pad_idx=self.padding_idx)
            elif k == 'src_charges':
                v = pad_1d_tokens([torch.tensor(s[0][k]).long() for s in samples], pad_idx=self.charge_padding_idx)
            batch[k] = v
        try:
            label = torch.tensor(np.array([s[1] for s in samples]))
        except:
            label = None
        return batch, label