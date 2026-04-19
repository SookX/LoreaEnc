import numpy as np
from dataclasses import dataclass, asdict
from typing import Literal
from typing import Optional
import torch 
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

@dataclass
class Wav2Vec2ForPreTrainingOutput:

    loss: Optional[torch.FloatTensor] = None
    projected_states: torch.FloatTensor = None
    projected_quantized_states: torch.FloatTensor = None
    codevector_perplexity: torch.FloatTensor = None
    contrastive_loss: Optional[torch.FloatTensor] = None
    diversity_loss: Optional[torch.FloatTensor] = None

@dataclass
class Wav2Vec2Config:

    ### FEATURE ENCODER CONVOLUTION CONFIG ###
    conv_dim: tuple = (512, 512, 512, 512, 512, 512, 512)
    conv_stride: tuple = (5, 2, 2, 2, 2, 2, 2)
    conv_kernel: tuple = (10, 3, 3, 3, 3, 2, 2)
    conv_bias: bool = True
    feature_projection_dropout_p: float = 0.0

    ### POSITIONAL CONVOLUTIONAL EMBEDDING ###
    conv_positional_emb_drop_p: float = 0.0
    conv_positional_emb_groups: int = 16
    conv_positional_emb_kernel_size: int = 128

    ### TRANSFORMER CONFIG ###
    num_transformer_layers: int = 12
    num_attention_heads: int = 12
    embedding_dimension: int = 768
    mlp_ratio: int = 4
    mlp_dropout_p: float = 0.0
    attention_dropout_p: float = 0.0
    transformer_encoder_dropout: float = 0.0
    layer_dropout: float = 0.0
    initializer_range: float = 0.02

    ### GUMBEL SOFTMAX CONFIG ###
    num_codevector_groups: int = 2
    num_codevectors_per_group: int = 320
    codevector_dim: int = 256
    pre_quantizer_dropout: float = 0.0

    ### MASKING CONFIG ###
    masking_probability: float = 0.065
    masking_span_length: int = 10 
    minimum_spans: int = 2

    ### LOSS CONFIG ###
    contrastive_logits_temperature: float = 0.1
    diversity_loss_weight: float = 0.1

    ### TRAINING CONFIG ###
    num_negatives: int = 100

    ### LayerNorm Config ###
    layer_norm_eps: float = 1e-5
    
    ### CTC Config ###
    asr_head_dropout_p: float = 0.1
    blank_token_idx: int = 0
    vocab_size: int = 32
    ### Huggingface Interface Config ###
    hf_model_name: str = "facebook/wav2vec2-base"

    ### Pretrain Backbone Config ###
    path_to_pretrained_weights: str = None

    ### Backbone Config ###
    pretrained_backbone: Literal["pretrained", "pretrained_huggingface", "random"] = "pretrained"

    ### Added in to_dict() method so this Config is compatible with Huggingface Trainer!!! ###
    def to_dict(self):
        return asdict(self)

def compute_encoded_lengths(lengths, conv_kernels, conv_strides):
    if  not isinstance(lengths, torch.Tensor):
        lengths = torch.Tensor(lengths)

    def _compute_conv_out(lenghts, kernel_size, stride):
        return torch.floor((lenghts - (kernel_size - 1) - 1) / stride) + 1 
    
    for k, s in zip(conv_kernels, conv_strides):
        lengths = _compute_conv_out(lengths, k, s)

    lengths = lengths.type(torch.int)
    return lengths
    
def compute_sub_attention_mask(config, attention_mask):

    batch_size = attention_mask.shape[0]

    raw_lengths = attention_mask.sum(axis=-1) # 1 for all valid
    
    encoded_lengths = compute_encoded_lengths(raw_lengths, config.conv_kernel, config.conv_stride)

    sub_attention_mask = torch.zeros((batch_size, max(encoded_lengths)))
    for idx, lengths in enumerate(encoded_lengths):
        sub_attention_mask[idx, :lengths] = 1

    return sub_attention_mask

def compute_span_mask(shape,
                      mask_prob=0.065,
                      mask_length=10,
                      min_masks = 2,
                      attention_mask=None):

    batch_size, total_sequence_length = shape

    if attention_mask is None:
        sequence_lengths = [total_sequence_length] * batch_size
    else:
        sequence_lengths = attention_mask.sum(axis=-1).to(torch.int).tolist()

    sequence_masks = []
    for length in sequence_lengths:
        mask = torch.zeros(total_sequence_length).bool()
        sample_starting_idx = (torch.rand(length) < mask_prob).nonzero()

        if len(sample_starting_idx) < min_masks:
            sample_starting_idx = torch.randint(low=0, high=length, size=(min_masks, 1)) # randomly sample 2 values for sampling points if nothing was selected
        
        span_offsets = torch.arange(mask_length)
        spans = sample_starting_idx + span_offsets

        spans = spans.flatten()

        spans = spans[spans <= length - 1]
        mask[spans.flatten()] = True
        
        sequence_masks.append(mask.unsqueeze(0))
        
    sequence_mask = torch.concatenate(sequence_masks)
    return sequence_mask

def sample_negative_indices(features_shape, num_negatives, mask_time_indices):

    """
    This is kind of finiky, im sure there is a better way to do this but here goes!

    What we need to do is for every masked sample, we need to generated 'num_negative' number of negative samples
    for our contrastive loss from the other masked samples. There are a few steps as far as I can tell to do this:
    
    We start with the masked_indexes, a boolean vector for each sequence indicating if that index is being masked.
     
    To be clear, there are two indexes going on:
        
        (1) Masked Index which are the actual indexes of the masked tokens in a sample of encoded/masked data -> [42, 58, 64, ...]
        (2) Enumerated Index which is from 0 to the number of masked tokens -> [0, 1, 2, ...]
    
    So, for every sample in the batch we need to:
    
        (1) Grab the masked indexes and number of masked tokens in the sample
        (2) Uniformly sample 'num_negative' number of enumerated indexes ranging from 0 to the number of masked tokens
        (3) We need to ensure that anything we sample is NOT THE POSITIVE SAMPLE. So if we are on enumerated index 0, and
            have 100 masked tokens, we can sample any enumerated index from 1 to 99, but NOT 0. So we check for this and then just add
            1 to any case. Therefore if there is an overlap where for index 0 we are sampling the negative 0, we will now
            just be sampling 1!
        (4) If we are adding 1 to everything, there is now a chance that our enumerated indexes may go above the number of mask tokens, 
            which also isn't good... So if we have 100 tokens, the only case this happens is if in the sampling for token 99, 
            we get the index 99 for the negative. This will make this value 100, as per the previous step. Therefore in this case
            we will just resample a new value for anything greater, with any value between 0 and 1 less than the max possible, in this case
            98. 
        (5) Use these computed enumerated indexes to index the original masked token indexes to get our negatives


    Whew... these arent very fun manipulations at all...

    Caveat: Not totally sure, but if we want 100 negatives, but only 50 were masked, there will be repeated negatives in this case, probably fine?

    Args:
        masked_indexes: Tensor of shape (Batch x Sequence Length) indicating the location of masked tokens
        num_negatives: Number of negatives we want to sample

    """


    ### Pass in the Data Shape (Post Convolutional Encoding) ###
    batch_size, sequence_length = features_shape
    
    ### Get Indexes for sequence of features ###
    sequence_index = np.arange(sequence_length)
    
    ### Empty Tensor to fill with sampled negatives ###
    sampled_negatives = np.zeros(shape=(batch_size, sequence_length, num_negatives), dtype=np.int32)
    
    ### Create Span Mask if not supplied (Nothing to sample though in this case) ###
    if mask_time_indices is None:
        mask_time_indices = np.ones((batch_size, sequence_length), dtype=bool)

    for idx in range(batch_size):
        
        ### Grab Span Mask for Sample ###
        batch_span_mask = mask_time_indices[idx]

        ### Grab the Corresponding Mask Index for Sequence in this Batch ###
        masked_indexes = sequence_index[batch_span_mask]

        ### Create Matrix of feature indices to avoid sampling positive tensor ###
        num_masked = batch_span_mask.sum()
        feature_index = np.expand_dims(np.arange(num_masked),-1)
        feature_index = np.repeat(feature_index, num_negatives, axis=-1)
   
        ### Sample Indicies (Notice, we will sample index 0 to num_masked - 1) ###
        ### This is so if there is an overlap between sampled index and the positive (true) index ###
        ### We can just add 1, but keep the highest index to num_masked ###
        sample_index = np.random.randint(0, num_masked-1, size=(num_masked, num_negatives))

        ### If our Sampled Index is Equal to our Feature index, that is a repeat of a positive class, so just add 1 to make it different! ###
        sample_index[(sample_index == feature_index)] += 1

        ### Store these Sample Indexes in our sampled_negatives array with the corresponding masked index ###
        ### Break this down: 
        ### sampled_negatives[idx] -> Index the batch dimension in our sampled_negatives (starts out as all zeros)
        ### sampled_negatives[idx][batch_span_mask] -> this indexes our batch of sampled_negatives only for the indexes that we have a mask (we only sample negatives for masked locations)
        ### masked_indexes: if our sequence length is 20, the masked_indexes is which indexes from 0 to 19 are masked
        ### sampled_index: if we have 8 masked things in our sequence then the sampled_index goes from 0 to 7. This tensor randomly samples those indexes to have num_negative negatives for each masked position
        ### masked_indexes[sample_index]: We dont care about the sampled_index from 0 to 7, we want the negative indexes in terms of the original index 0 to 19, so this converts the sampled index to our sequence index!
        sampled_negatives[idx][batch_span_mask] = masked_indexes[sample_index]

        ### In the future, we will flatten all this, so if we have a sequence length of 20, the first sample should go from 0 to 19, but the second sample should go from 20 to 39. 
        ### So we need to just adjust for batch size. Everything starts at the 0 index right now, so we just ned to add what sample in the batch are we on, times the sequence length 
        sampled_negatives[idx] += idx * sequence_length

    ### Convert to PyTorch Tensor ###
    sampled_negatives = torch.tensor(sampled_negatives, dtype=torch.long)

    return sampled_negatives

if __name__ == "__main__":

    ### Prepare some Random Data ###
    seq_lens = [25000, 32000]
    
    data = [torch.randn(l) for l in seq_lens]
    attention_mask = [torch.ones(l) for l in seq_lens]

    data = torch.nn.utils.rnn.pad_sequence(data, padding_value=0.0, batch_first=True)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, padding_value=0, batch_first=True)

    ### Test Span Masking / Sampling Pipeline ###
    config = Wav2Vec2Config()
    
    sub_attention_mask = compute_sub_attention_mask(config, attention_mask)
    span_mask = compute_span_mask(shape=tuple(sub_attention_mask.shape),
                                  attention_mask=sub_attention_mask)
    negatives = sample_negative_indices(features_shape=tuple(sub_attention_mask.shape), 
                                        num_negatives=5,
                                        mask_time_indices=span_mask)
    