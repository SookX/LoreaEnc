
import torch


def collate_fn(batch):

    """
    This collate function is basically the heart of our implementation! It includes everything we need for training
    such as attention masks, sub_attention_masks, span_masks and our sampled negatives!
    """

    ### Sort Batch from Longest to Shortest (for future packed padding) ###
    batch = sorted(batch, key=lambda x: x["input_values"].shape[0], reverse=True)
    
    ### Grab Audios from our Batch Dictionary ###
    batch_mels = [sample["input_values"] for sample in batch]
    batch_transcripts = [sample["labels"] for sample in batch]
    raw_audios = [sample["raw_audio"].squeeze(0).detach().cpu().numpy() for sample in batch]
    raw_transcripts = [sample["raw_transcript"] for sample in batch]
     # list of 1D numpy arrays
    teacher_logits = None
    try:
        teacher_logits = [sample["teacher_logits"] for sample in batch]
    except:
        pass

   # raw_audios = [a.numpy() for a in batch["raw_audios"]]  # list of 1D numpy arrays


    ### Get Length of Audios ###
    seq_lens = torch.tensor([b.shape[0] for b in batch_mels], dtype=torch.long)

    ### Pad and Stack Spectrograms ###
    spectrograms = torch.nn.utils.rnn.pad_sequence(batch_mels, batch_first=True, padding_value=0)

    ### Convert to Shape Convolution Is Happy With (B x C x H x W) ###
    spectrograms = spectrograms.transpose(-1,-2)

    ### Get Target Lengths ###
    target_lengths = torch.tensor([len(t) for t in batch_transcripts], dtype=torch.long)

    ### Pack Transcripts (CTC Loss Can Take Packed Targets) ###
    packed_transcripts = torch.cat(batch_transcripts)
    
    ### Create Batch ###
    batch = {"input_values": spectrograms, 
             "seq_lens": seq_lens, 
             "labels": packed_transcripts, 
             "target_lengths": target_lengths,
             "raw_audios": raw_audios,
             "teacher_logits": teacher_logits,
             "raw_transcripts": raw_transcripts}
    
    return batch


