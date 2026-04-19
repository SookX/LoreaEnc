import os
import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model as HFWav2Vec2Model
from safetensors.torch import load_file
from utils import (
    Wav2Vec2ForPreTrainingOutput,
    compute_sub_attention_mask,
    compute_encoded_lengths
)
class Wav2Vec2LayerNormConvLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 bias):
        
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias
        )

        self.layernorm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        # (B x C x L)
        x = self.conv(x)

        # (B x L x C)
        x = x.transpose(-1, -2)
        x = self.layernorm(x)
        x = x.transpose(-1, -2)

        x = self.activation(x)

        return x

class Wav2Vec2FeatureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        assert len(config.conv_dim) == len(config.conv_stride) == len(config.conv_kernel), \
            "Check Config for same number of convolution components"
        
        num_conv_blocks = len(config.conv_kernel)

        conv_channels = (1, ) + tuple(config.conv_dim)
        
        self.conv_layers = nn.ModuleList()
        for conv_idx in range(num_conv_blocks):
            self.conv_layers.append(
                Wav2Vec2LayerNormConvLayer(in_channels=conv_channels[conv_idx],
                        out_channels=conv_channels[conv_idx + 1],
                        kernel_size=config.conv_kernel[conv_idx],
                        stride=config.conv_stride[conv_idx],
                        bias=config.conv_bias)
            )
    
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        
        return x

class Wav2Vec2PositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv = nn.Conv1d(
            config.embedding_dimension,
            config.embedding_dimension,
            kernel_size=config.conv_positional_emb_kernel_size,
            padding=config.conv_positional_emb_kernel_size // 2,
            groups=config.conv_positional_emb_groups
        )

        self.activation = nn.GELU()

    def forward(self, x):
        
        batch_size, seq_len, embed = x.shape
        x = x.transpose(1, 2)

        positional_embeddings = self.conv(x)
        positional_embeddings = positional_embeddings[:, :, :seq_len]
        positional_embeddings = self.activation(positional_embeddings)
        positional_embeddings = positional_embeddings.transpose(1, 2)

        return positional_embeddings

class Wav2Vec2FeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.projection = nn.Linear(config.conv_dim[-1], config.embedding_dimension)
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1])
        self.dropout = nn.Dropout(config.feature_projection_dropout_p)

    def forward(self, x):
        normed_x = self.layer_norm(x)
        projected_x = self.projection(normed_x)
        projected_x = self.dropout(projected_x)

        return projected_x, normed_x
        
class Wav2Vec2Attention(nn.Module):
    """
    Regular Self-Attention but in this case we utilize flash_attention
    incorporated in the F.scaled_dot_product_attention to speed up our training. 
    """
    def __init__(self, config):
        super(Wav2Vec2Attention, self).__init__()
        
        ### Store Config ###
        self.config = config
        
        ### Sanity Checks ###
        assert config.embedding_dimension % config.num_attention_heads == 0, "Double check embedding dim divisible by number of heads"

        ### Attention Head Dim ###
        self.head_dim = config.embedding_dimension // config.num_attention_heads

        ### Attention Projections ###
        self.q_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.k_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.v_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)

        ### Post Attention Projection ###
        self.out_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        

    def forward(self, x, attention_mask=None):

        ### Store Shape ###
        batch, seq_len, embed_dim = x.shape

        ### Compute Attention with Flash Attention ###
        q = self.q_proj(x).reshape(batch, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1,2).contiguous()
        k = self.k_proj(x).reshape(batch, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1,2).contiguous()
        v = self.v_proj(x).reshape(batch, seq_len, self.config.num_attention_heads, self.head_dim).transpose(1,2).contiguous()
        
        ### Compute Attention (Attention Mask has shape Batch x Sequence len x Sequence len) ###
        attention_out = F.scaled_dot_product_attention(q, k, v, 
                                                        attn_mask=attention_mask, 
                                                        dropout_p=self.config.attention_dropout_p if self.training else 0.0)


        ### Compute Output Projection ###
        attention_out = attention_out.transpose(1,2).flatten(2)
        attention_out = self.out_proj(attention_out)

        return attention_out
    
class Wav2Vec2FeedForward(nn.Module):
    """
    Regular MLP module after our attention computation. 
    """
    def __init__(self, config):
        super(Wav2Vec2FeedForward, self).__init__()
        
        hidden_size = config.embedding_dimension * config.mlp_ratio
        self.intermediate_dense = nn.Linear(config.embedding_dimension, hidden_size)
        self.activation = nn.GELU()
        self.intermediate_dropout = nn.Dropout(config.mlp_dropout_p)

        self.output_dense = nn.Linear(hidden_size, config.embedding_dimension)
        self.output_dropout = nn.Dropout(config.mlp_dropout_p)

    def forward(self, x):
        x = self.intermediate_dense(x)
        x = self.activation(x)
        x = self.intermediate_dropout(x)

        x = self.output_dense(x)
        x = self.output_dropout(x)
        return x

class Wav2Vec2EncoderLayer(nn.Module):
    """
    Single transformer block stacking together Attention and our FeedForward
    layers, with normalization and residual connections. 
    """
    def __init__(self, config):
        super(Wav2Vec2EncoderLayer, self).__init__()

        self.attention = Wav2Vec2Attention(config)
        self.dropout = nn.Dropout(config.transformer_encoder_dropout)
        self.layer_norm = nn.LayerNorm(config.embedding_dimension, eps=config.layer_norm_eps)
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.embedding_dimension, eps=config.layer_norm_eps)

    def forward(self, x, attention_mask=None):

        x = x + self.dropout(self.attention(x, attention_mask=attention_mask))
        x = self.layer_norm(x)

        x = x + self.feed_forward(x)
        x = self.final_layer_norm(x)

        return x
    
class Wav2Vec2Encoder(nn.Module):
    """
    Convolutional positional embeddings followed by a stack of transformer blocks.
    The sub_attention_mask passed in is (Batch x Sequence), where False = padding.
    """
    def __init__(self, config):
        super(Wav2Vec2Encoder, self).__init__()

        self.config = config

        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.embedding_dimension, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.conv_positional_emb_drop_p)

        self.layers = nn.ModuleList(
            [Wav2Vec2EncoderLayer(config) for _ in range(config.num_transformer_layers)]
        )

    def forward(self, x, attention_mask=None):

        batch_size, seq_len, embed_dim = x.shape

        if attention_mask is not None:
            attention_mask = attention_mask.bool()

            # Zero out padding positions so they don't pollute the residual stream
            x[~attention_mask] = 0

            # scaled_dot_product_attention expects (B, 1, seq_len, seq_len).
            # expand() is a zero-copy view; repeat() would allocate a full copy.
            attention_mask = (
                attention_mask
                .unsqueeze(1)       # (B, 1, seq_len)
                .unsqueeze(1)       # (B, 1, 1, seq_len)
                .expand(batch_size, 1, seq_len, seq_len)  # broadcast, no copy
            )

        position_embeddings = self.pos_conv_embed(x)
        x = x + position_embeddings
        x = self.layer_norm(x)
        x = self.dropout(x)

        for layer in self.layers:
            dropout_probability = torch.rand(1)
            if (not self.training) or (dropout_probability >= self.config.layer_dropout):
                x = layer(x, attention_mask=attention_mask)

        return x

class Wav2Vec2Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.feature_extractor = Wav2Vec2FeatureEncoder(config)
        self.feature_projection = Wav2Vec2FeatureProjection(config)

        if config.masking_probability > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.embedding_dimension))
            torch.nn.init.uniform_(self.masked_spec_embed)

        self.encoder = Wav2Vec2Encoder(config)
    def forward(self,
            input_values,
            attention_mask = None,
            sub_attention_mask = None,
            mask_time_indices = None, 
            return_features_to_quantize=False):
        
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if (sub_attention_mask is None and attention_mask is not None):
            sub_attention_mask = compute_sub_attention_mask(self.config, attention_mask).to(input_values.device)
        
        hidden_states, extract_features = self.feature_projection(extract_features)

        hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        
        encoder_outputs = self.encoder(hidden_states, attention_mask=sub_attention_mask)

        if return_features_to_quantize:
            return encoder_outputs, extract_features
        
        else:
            return encoder_outputs

class Wav2Vec2ForPreTraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout_features = nn.Dropout(config.pre_quantizer_dropout)
        self.quantizer = Wav2Vec2GumbelVectorQuantizer(config)
        self.project_hid = nn.Linear(config.embedding_dimension, config.codevector_dim)
        self.project_q = nn.Linear(config.codevector_dim, config.codevector_dim)
        self.apply(weight_init_strategy(config))

    def set_gumbel_temperature(self, temperature):
        self.quantizer.temperature = temperature

    def _compute_cosine_similarity(self, target_features,
                                   negative_features, 
                                   predicted_features, 
                                   temperature=0.1):
        
        ### Just a reminder of tensor shapes:
        ### true_quantized: (Batch x Sequence Length x VQ_dim)
        ### negative_quantized: (Num Negatives x Batch Size x Sequence Length x VQ_dim)
        ### transformer_output: (Batch x Sequence Length x VQ_dim)

        ### So, what we want to do is compute the cosine similarity between each token in the transformer output
        ### against the Num Negatives + 1 Positive quantized tokens! So lets first concatenate the true quantized
        ### to our negatives. To do this, we need to add a dimension to our true_quantized features so its becomes
        ### (1 x Batch x Sequence Length x VQ_dim). This will create our quantized targets in the shape of 
        ### (Num_negatives + 1 x Batch x Sequence Length x VQ_dim)
        target_features = target_features.unsqueeze(0)
        targets = torch.cat([target_features, negative_features], dim=0) 

        ### Compute Cosine Similarity between our transformer output and targets, along the VQ_dimension (which is the last one) ###
        ### If you take a quick look at the PyTorch cosine similarity function (https://pytorch.org/docs/stable/generated/torch.nn.functional.cosine_similarity.html)
        ### we just needt to make sure our shapes are broadcastable to a common shape, but we already made this happen because our
        ### transformer output is (batch x sequence x vq_dim) and the targets are (Num_negatives + 1 x Batch x Sequence Length x VQ_dim)
        ### So each sample in our Batch x sequence will compute its cosine similarity across all Num negatives + 1 quantized tokens. 
        ### This operation will return the cosine similarity in the shape of (Num_negatives + 1 x Batch x Sequence Length)
        cosine_sim = torch.cosine_similarity(predicted_features, targets, dim=-1)

        ### Now in formula 4, we see that there is a softmax involved. We can actually reformulate this problem easily in terms of CrossEntropyLoss that I explain below ###
        ### Torch CrossEntropyLoss expects logits, so we wont do any softmax now. But, we can scale our cosine similarity by our temperature parameter (kappa in the formula)! 
        cosine_sim = cosine_sim / temperature

        return cosine_sim


    def forward(self,
                input_values,
                attention_mask = None,
                sub_attention_mask = None,
                mask_time_indices = None, 
                sampled_negative_indices=None):

        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)
        
        transformer_outputs, features_to_quantize = self.wav2vec2(input_values,
                                                                  attention_mask,
                                                                  sub_attention_mask,
                                                                  mask_time_indices,
                                                                  return_features_to_quantize = True)
        
        quantized_codes, perplexity = self.quantizer(features_to_quantize, mask_time_indices)
        
        quantized_codes = self.project_q(quantized_codes)
        transformer_vq = self.project_hid(transformer_outputs)

        loss = None
        diversity_loss = None
        contrastive_loss = None

        if sampled_negative_indices is not None:
            batch_size, seq_len, vq_size = quantized_codes.shape
            _, _, num_negatives = sampled_negative_indices.shape
            
            negative_quantized_codes = quantized_codes.reshape(-1, vq_size)[sampled_negative_indices.flatten()]
            negative_quantized_codes = negative_quantized_codes.reshape(batch_size, seq_len, num_negatives, vq_size).permute(2, 0, 1, 3)
            cosine_sim = self._compute_cosine_similarity(target_features=quantized_codes,
                                                         negative_features=negative_quantized_codes,
                                                         predicted_features=transformer_vq,
                                                         temperature=self.config.contrastive_logits_temperature)
            
            neg_equals_pos_mask = (quantized_codes == negative_quantized_codes).all(dim=-1)
            if neg_equals_pos_mask.any():
                cosine_sim[1:][neg_equals_pos_mask] = float("-inf")
            
            cosine_sim = cosine_sim.permute(1, 2, 0).reshape(batch_size * seq_len, num_negatives + 1)
            labels = torch.ones(len(cosine_sim), device=cosine_sim.device, dtype=torch.long) * -100 # ignore loss
            labels[mask_time_indices.flatten()] = 0 # The correct is always the first
            contrastive_loss = F.cross_entropy(cosine_sim, labels, reduction="sum")

            GV = self.config.num_codevector_groups * self.config.num_codevectors_per_group
            diversity_loss = (((GV - perplexity)) / GV) * mask_time_indices.sum()

            loss = contrastive_loss + self.config.diversity_loss_weight *  diversity_loss

        return Wav2Vec2ForPreTrainingOutput(
                loss=loss,
                projected_states=transformer_outputs,
                projected_quantized_states=quantized_codes,
                codevector_perplexity=perplexity,
                contrastive_loss=contrastive_loss,
                diversity_loss=diversity_loss,
            )

class Wav2Vec2GumbelVectorQuantizer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_groups = config.num_codevector_groups
        self.num_vars = config.num_codevectors_per_group

        self.codevectors = nn.Parameter(
            torch.FloatTensor(1, self.num_groups * self.num_vars, config.codevector_dim // self.num_groups)
        )
        self.weight_proj = nn.Linear(config.conv_dim[-1], self.num_groups * self.num_vars)
        self.temperature = 2

    def _compute_perplexity(self, probs, mask=None):
        
        if mask is not None:

            marginal_probs = probs[mask.flatten()]
            marginal_probs = marginal_probs.sum(dim=0) / mask.sum()
        
        else:
            marginal_probs = probs.mean(0)    
        
        perplexity_per_codebook = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7),  dim=-1))

        perplexity = perplexity_per_codebook.sum()

        return perplexity

    def forward(self, hidden_states, mask_time_indices = None):
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.reshape(batch_size * seq_len * self.num_groups, -1)
        
        if self.training:
            codevector_probs = nn.functional.gumbel_softmax(hidden_states.float(), tau=self.temperature, hard=True)

            ## Compute perplexity ##
            hidden_states = hidden_states.reshape(batch_size * seq_len, self.num_groups, -1)
            codevector_soft_dist = hidden_states.softmax(axis=-1)
            perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)

        else:
            codevector_idx = hidden_states.argmax(dim=-1)
            codevector_probs = torch.zeros_like(hidden_states)
            codevector_probs[torch.arange(hidden_states.shape[0]), codevector_idx] = 1
            codevector_probs = codevector_probs.reshape(batch_size * seq_len, self.num_groups, -1)
            perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)
        
        codevector_probs = codevector_probs.reshape(batch_size * seq_len, -1)
        
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors.type_as(codevector_probs)
        
        codevectors = codevectors_per_group.reshape(batch_size * seq_len, self.num_groups, self.num_vars, -1)
        
        codevectors = codevectors.sum(dim=-2)
        
        codevectors = codevectors.reshape(batch_size, seq_len, -1)
        
        return codevectors, perplexity

class Wav2Vec2ForCTC(nn.Module):

    """
    The toughest part for Automatic Speech Recognition is learning the alignment between 
    two arbritarily long sequences: The raw audio and the characters in the sentence. CTCLoss
    is one such method that learns this alignment so we can actually do ASR. 

    This model can be initialized with our own pretrained backbone, but if you dont have a ton of
    GPU resources or time, you can use the Huggingface backbone as well!
    """

    def __init__(self, config):
        super().__init__()
        
        self.config = config

        ### Grab Backbone Based on Config ###
        self.load_backbone()

        ### Initialize Prediction Head ###
        self.dropout = nn.Dropout(config.asr_head_dropout_p)
        self.lm_head = nn.Linear(config.embedding_dimension, config.vocab_size)

    def load_backbone(self):
        
        if self.config.pretrained_backbone == "pretrained_huggingface":
            print(f"Loading Huggingface Wav2Vec2 Backbone: {self.config.hf_model_name}")
            self.wav2vec2 = HFWav2Vec2Model.from_pretrained(self.config.hf_model_name)
        else:
            self.wav2vec2 = Wav2Vec2Model(self.config)

            if self.config.pretrained_backbone == "pretrained":
                if self.config.path_to_pretrained_weights is None:
                    raise Exception("Provide the argument `path_to_pretrained_weights` in the config, else we cant load them!")
                else:
                    
                    if not os.path.isfile(self.config.path_to_pretrained_weights):
                        raise Exception(f"Provided path to safetensors weights {self.config.path_to_pretrained_weights} is invalid!")

                    print(f"Loading Wav2Vec2Model Backbone from {self.config.path_to_pretrained_weights}")

                    ### Load Weights with load_file from safetensors ###
                    state_dict = load_file(self.config.path_to_pretrained_weights)

                    ### Cleanup of Weights and keys ###
                    backbone_keys = {}
                    for key in state_dict.keys():

                        ### If Wav2Vec2 is in key name, just remove from the key name ###
                        if "wav2vec2" in key:
                            new_key = key.replace("wav2vec2.", "")
                            backbone_keys[new_key] = state_dict[key]

                        ### If wav2vec2 is not in key name, it isnt a part of the backbone so ignore it ###
                        else:
                            continue

                    ### Load State Dict to Backbone ###
                    self.wav2vec2.load_state_dict(backbone_keys)

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        print("Freezing Convolutional Feature Encoder")
        if self.config.pretrained_backbone == "pretrained_huggingface":
            ### Huggingface already has a method to freeze model parameters ###
            self.wav2vec2.feature_extractor._freeze_parameters()
        elif self.config.pretrained_backbone == "pretrained":
            for param in self.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False
        elif self.config.pretrained_backbone == "random":
            raise Exception("Feature Encoder is Randomly initialized!!! You are disabling gradients, you need to train them!")
        else:
            raise ValueError(f"Inputed pretrained_backbone {self.config.pretrained_backbone} not in (pretrained, pretrained_huggingface, random)")
        
    def forward(
        self,
        input_values,
        attention_mask = None,
        labels = None):

        ### Our Pretrained model and Huggingface Wav2Vec2Model have slightly different forwards and returns ###
        if self.config.pretrained_backbone == "pretrained_huggingface":
            outputs = self.wav2vec2(
                input_values,
                attention_mask=attention_mask,
            )

            hidden_states = outputs.last_hidden_state
        
        else:

            hidden_states = self.wav2vec2(input_values, 
                                          attention_mask, 
                                          return_features_to_quantize=False)

        ### Pass through Dropout and Compute Logits ### 
        hidden_states = self.dropout(hidden_states)
        logits = self.lm_head(hidden_states)

        ### If labels are provided (already tokenized) we can compute our CTC Loss as well ###
        loss = None
        if labels is not None:

            ### If our Attention Mask is None, then attend to all tokens ###
            if attention_mask is None:
                attention_mask = torch.ones_like(input_values, dtype=torch.long)

            ### Compute Input Sizes of feature extracted audio via sub_attention_mask ###
            input_lengths = compute_encoded_lengths(attention_mask.sum(-1), self.config.conv_kernel, self.config.conv_stride).to(torch.long)

            ### Labels are -100 for padding tokens (as per our collate function), no need to keep for loss ###
            labels_mask = (labels >= 0)

            ### Add up nonpad tokens to see the number of tokens per sequence in batch for target sizes ###
            target_lengths = labels_mask.sum(-1)

            ### Grab nonpadded labels (CTC Loss can take flatten vector of (unpadded) inputs of shape (sum(target_lengths)) ###
            flattened_targets = labels.masked_select(labels_mask)

            ### CTC Loss takes log probs and doesnt work in mixed precision, make sure its float32 ###
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32)

            ### CTC Loss expects (Sequence x Batch x Vocab Size) but we have (Batch x Sequence x Vocab size) ###
            log_probs = log_probs.transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=0,
                    reduction="mean",
                    zero_infinity=False,
                )

        return loss, logits

def weight_init_strategy(config):

    def _init_weights(module):

        if isinstance(module, Wav2Vec2ForPreTraining):
            module.project_hid.reset_parameters()
            module.project_q.reset_parameters()

        elif isinstance(module, Wav2Vec2GumbelVectorQuantizer):
            module.weight_proj.weight.data.normal_(mean=0.0, std=1)
            module.weight_proj.bias.data.zero_()
            nn.init.uniform_(module.codevectors)

        elif isinstance(module, Wav2Vec2PositionalConvEmbedding):
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            nn.init.constant_(module.conv.bias, 0)

        elif isinstance(module, Wav2Vec2FeatureProjection):
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)

        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)
    
    return _init_weights

if __name__ == "__main__":

    from utils import Wav2Vec2Config
    from dataset import LibriSpeechDataset, Wav2Vec2CollateFunctionForPreTraining
    from torch.utils.data import DataLoader

    w2v2_config = Wav2Vec2Config(num_transformer_layers=2)
    model = Wav2Vec2ForPreTraining(config=w2v2_config)
    #model = Wav2Vec2Model(w2v2_config)
    dataset = LibriSpeechDataset(path_to_data_root="./dataset/", 
                                 include_splits=["dev-clean"])
    collate_fn = Wav2Vec2CollateFunctionForPreTraining(config=w2v2_config)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    batch = next(iter(loader))
    #print(batch["input_values"].shape)
    # model_inputs = {k: v for k, v in batch.items() if k != "sampled_negative_indices"}
    out = model(**batch)