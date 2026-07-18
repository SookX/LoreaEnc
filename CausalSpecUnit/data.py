import os
import sys
import time
import json
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset


def dataset_trace(message):
    if os.environ.get("CSU_TRACE_DATASET", "0") != "1":
        return
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    rank = os.environ.get("RANK", "?")
    print(f"[data-trace {now}] rank={rank} {message}", file=sys.stderr, flush=True)


def iter_librispeech_items(data_root, splits):
    for split in splits:
        split_dir = os.path.join(data_root, split)
        for speaker in sorted(os.listdir(split_dir)):
            speaker_dir = os.path.join(split_dir, speaker)
            if not os.path.isdir(speaker_dir):
                continue
            for chapter in sorted(os.listdir(speaker_dir)):
                chapter_dir = os.path.join(speaker_dir, chapter)
                if not os.path.isdir(chapter_dir):
                    continue
                txt_files = [p for p in os.listdir(chapter_dir) if p.endswith(".txt")]
                if not txt_files:
                    continue
                with open(os.path.join(chapter_dir, txt_files[0]), encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        uid = parts[0]
                        transcript = " ".join(parts[1:])
                        yield {
                            "uid": uid,
                            "audio_path": os.path.join(chapter_dir, uid + ".flac"),
                            "transcript": transcript,
                            "split": split,
                        }


def iter_mls_items(lang_root, splits):
    """Yield item dicts for Multilingual LibriSpeech, matching iter_librispeech_items.

    lang_root: a single-language MLS root, e.g. ``dataset/mls/mls_polish``.
    splits:    subset of {"train", "dev", "test"}.

    MLS layout::

        <lang_root>/<split>/transcripts.txt      # "<uid>\\t<transcript>" per line
        <lang_root>/<split>/audio/<spk>/<book>/<uid>.flac   # uid = "<spk>_<book>_<utt>"

    Audio is already 16 kHz mono flac, so no resampling is needed downstream.
    """
    for split in splits:
        split_dir = os.path.join(lang_root, split)
        trans_path = os.path.join(split_dir, "transcripts.txt")
        if not os.path.isfile(trans_path):
            raise FileNotFoundError(f"MLS transcripts not found: {trans_path}")
        with open(trans_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    continue
                uid, transcript = parts[0], parts[1].strip()
                if not transcript:
                    continue
                uid_parts = uid.split("_")
                if len(uid_parts) < 3:
                    continue
                spk, book = uid_parts[0], uid_parts[1]
                yield {
                    "uid": uid,
                    "audio_path": os.path.join(split_dir, "audio", spk, book, uid + ".flac"),
                    "transcript": transcript,
                    "split": split,
                }


def enumerate_items(data_root, splits, mls_lang_root=None):
    """Enumerate utterances: LibriSpeech by default, or a single-language MLS
    root when ``mls_lang_root`` is set (e.g. ``dataset/mls/mls_polish``)."""
    if mls_lang_root:
        return list(iter_mls_items(mls_lang_root, splits))
    return list(iter_librispeech_items(data_root, splits))


def audio_duration_seconds(path):
    info = torchaudio.info(path)
    return float(info.num_frames) / float(info.sample_rate)


def subset_items_by_hours(items, max_hours, seed=42):
    if max_hours is None:
        return items, None
    if max_hours <= 0:
        raise ValueError(f"max_hours must be positive, got {max_hours}")

    rng = random.Random(seed)
    candidates = list(items)
    rng.shuffle(candidates)
    selected = []
    total_seconds = 0.0
    target_seconds = float(max_hours) * 3600.0

    for item in candidates:
        try:
            duration = audio_duration_seconds(item["audio_path"])
        except Exception as exc:
            dataset_trace(f"duration read failed uid={item['uid']} path={item['audio_path']} error={exc}")
            continue
        item = dict(item)
        item["duration_seconds"] = duration
        selected.append(item)
        total_seconds += duration
        if total_seconds >= target_seconds:
            break

    if not selected:
        raise RuntimeError(
            f"No utterances selected for max_hours={max_hours}. "
            "Check that the LibriSpeech audio files exist and torchaudio can read metadata."
        )
    return selected, total_seconds / 3600.0


class LogMelExtractor(torch.nn.Module):
    def __init__(self, sample_rate=16000, n_mels=80):
        super().__init__()
        self.sample_rate = sample_rate
        self.mels = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=int(sample_rate * 0.025),
            hop_length=int(sample_rate * 0.010),
            f_min=0.0,
            f_max=sample_rate / 2,
            n_mels=n_mels,
            center=True,
            pad_mode="reflect",
            power=2.0,
        )
        self.amp2db = T.AmplitudeToDB(top_db=80.0)

    def load_audio(self, path):
        try:
            audio, sr = torchaudio.load(path, normalize=True)
        except Exception:
            import soundfile as sf
            wav, sr = sf.read(path, dtype="float32", always_2d=True)
            audio = torch.from_numpy(wav.T)
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.sample_rate)
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        audio = audio / (audio.abs().amax(dim=-1, keepdim=True) + 1e-9)
        audio = torch.cat([audio[:, :1], audio[:, 1:] - 0.97 * audio[:, :-1]], dim=-1)
        return audio

    @torch.no_grad()
    def forward(self, path):
        audio = self.load_audio(path)
        mel = self.amp2db(self.mels(audio))[0].T
        return mel.float()


def load_cmvn(path):
    state = torch.load(path, map_location="cpu")
    return state["mean"].float(), state["std"].float().clamp_min(1e-5)


def apply_cmvn(mel, mean, std):
    return (mel - mean.to(mel.device)) / std.to(mel.device)


def load_targets(targets_path):
    targets_dir = os.path.dirname(targets_path)
    index_path = os.path.join(targets_dir, "target_index.json")
    if not os.path.isfile(index_path):
        start = time.time()
        dataset_trace(f"monolithic targets load start path={targets_path}")
        targets = torch.load(targets_path, map_location="cpu")
        dataset_trace(f"monolithic targets load done entries={len(targets)} seconds={time.time() - start:.1f}")
        return targets

    start = time.time()
    dataset_trace(f"target index load start path={index_path}")
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    uid_to_shard = index["uid_to_shard"]
    shards_dir = os.path.join(targets_dir, index.get("shards_dir", "targets_shards"))
    targets = {}
    for shard_name in sorted(set(uid_to_shard.values())):
        shard_path = os.path.join(shards_dir, shard_name)
        dataset_trace(f"target shard load start path={shard_path}")
        shard = torch.load(shard_path, map_location="cpu")
        targets.update(shard)
    dataset_trace(f"sharded targets load done entries={len(targets)} seconds={time.time() - start:.1f}")
    return targets


class SpecUnitDataset(Dataset):
    def __init__(self, data_root, splits, targets_path, cmvn_path, max_items=None, mel_cache_dir=None,
                 mls_lang_root=None):
        start = time.time()
        dataset_trace(f"scan start data_root={data_root} splits={splits} mls_lang_root={mls_lang_root}")
        self.items = enumerate_items(data_root, splits, mls_lang_root)
        dataset_trace(f"scan done items={len(self.items)} seconds={time.time() - start:.1f}")
        if max_items is not None:
            self.items = self.items[:max_items]
        start = time.time()
        dataset_trace(f"targets load start path={targets_path}")
        self.targets = load_targets(targets_path)
        dataset_trace(f"targets load done entries={len(self.targets)} seconds={time.time() - start:.1f}")
        start = time.time()
        dataset_trace(f"cmvn load start path={cmvn_path}")
        self.mean, self.std = load_cmvn(cmvn_path)
        dataset_trace(f"cmvn load done seconds={time.time() - start:.1f}")
        self.extractor = LogMelExtractor()
        self.mel_cache_dir = mel_cache_dir
        start = time.time()
        dataset_trace("filter start")
        self.items = [item for item in self.items if item["uid"] in self.targets]
        if self.mel_cache_dir:
            self.items = [
                item for item in self.items
                if os.path.isfile(os.path.join(self.mel_cache_dir, item["split"], item["uid"] + ".pt"))
            ]
        dataset_trace(f"filter done items={len(self.items)} seconds={time.time() - start:.1f}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        if self.mel_cache_dir:
            mel = torch.load(
                os.path.join(self.mel_cache_dir, item["split"], item["uid"] + ".pt"),
                map_location="cpu",
            ).float()
        else:
            mel = apply_cmvn(self.extractor(item["audio_path"]), self.mean, self.std)
        target = self.targets[item["uid"]]
        return {
            "uid": item["uid"],
            "mel": mel,
            "z100": target["z100"].long(),
            "z500": target["z500"].long(),
        }


class CTCSpecDataset(Dataset):
    """LibriSpeech CTC dataset.

    ssl_targets_path: when provided, also loads the precomputed K=100 / K=500
    SSL cluster targets from that path (or directory containing ``targets.pt``
    and the sharded index). Items without targets are filtered out — this is
    safe because the user's 10h/100h subsets are subsets of the 960h utterances
    those targets were generated for. Used by SSL-anchored fine-tuning.
    """
    def __init__(
        self,
        data_root,
        splits,
        tokenizer,
        cmvn_path=None,
        train_split=True,
        max_items=None,
        max_hours=None,
        subset_seed=42,
        ssl_targets_path=None,
        validate_audio=False,
        mls_lang_root=None,
    ):
        self.items = enumerate_items(data_root, splits, mls_lang_root)
        if max_items is not None:
            self.items = self.items[:max_items]
        self.audio_hours = None
        if max_hours is not None:
            self.items, self.audio_hours = subset_items_by_hours(self.items, max_hours, seed=subset_seed)
        self.tokenizer = tokenizer
        self.extractor = LogMelExtractor()
        self.train_split = train_split
        self.mean = self.std = None
        if cmvn_path is not None:
            self.mean, self.std = load_cmvn(cmvn_path)
        if validate_audio:
            # Pre-filter items whose audio cannot be decoded (corrupt FLACs,
            # missing files, codec mismatches). Uses torchaudio.info — header-only,
            # fast (~ms per file). Recommended for eval datasets where a single
            # bad file would otherwise crash a multi-hour run.
            kept, skipped = [], []
            for it in self.items:
                try:
                    torchaudio.info(it["audio_path"])
                    kept.append(it)
                except Exception as exc:
                    skipped.append((it["uid"], str(exc)[:80]))
            if skipped:
                print(
                    f"[CTCSpecDataset] filtered {len(skipped)} unreadable audio file(s) "
                    f"from {','.join(splits)}", flush=True,
                )
                for uid, err in skipped[:5]:
                    print(f"  - {uid}: {err}", flush=True)
                if len(skipped) > 5:
                    print(f"  ... and {len(skipped) - 5} more", flush=True)
            self.items = kept
        self.ssl_targets = None
        if ssl_targets_path is not None:
            # Accept either a directory (containing targets.pt + shards) or
            # the targets.pt file directly. load_targets handles both.
            targets_file = (
                os.path.join(ssl_targets_path, "targets.pt")
                if os.path.isdir(ssl_targets_path) else ssl_targets_path
            )
            self.ssl_targets = load_targets(targets_file)
            before = len(self.items)
            self.items = [it for it in self.items if it["uid"] in self.ssl_targets]
            if before != len(self.items):
                print(
                    f"[CTCSpecDataset] filtered {before - len(self.items)} items without "
                    f"SSL targets (kept {len(self.items)}/{before})", flush=True,
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        mel = self.extractor(item["audio_path"])
        if self.mean is not None:
            mel = apply_cmvn(mel, self.mean, self.std)
        labels = torch.tensor(self.tokenizer.encode(item["transcript"]), dtype=torch.long)
        out = {
            "uid": item["uid"],
            "mel": mel,
            "labels": labels,
            "transcript": item["transcript"],
        }
        if self.ssl_targets is not None:
            tgt = self.ssl_targets[item["uid"]]
            out["z100"] = tgt["z100"].long()
            out["z500"] = tgt["z500"].long()
        return out


def collate_ssl(batch):
    mel = nn.utils.rnn.pad_sequence([b["mel"] for b in batch], batch_first=True)
    lengths = torch.tensor([b["mel"].size(0) for b in batch], dtype=torch.long)
    z100 = nn.utils.rnn.pad_sequence([b["z100"] for b in batch], batch_first=True, padding_value=-100)
    z500 = nn.utils.rnn.pad_sequence([b["z500"] for b in batch], batch_first=True, padding_value=-100)
    target_lengths = torch.tensor([b["z100"].numel() for b in batch], dtype=torch.long)
    return mel, lengths, z100, z500, target_lengths


def collate_ctc(batch):
    mel = nn.utils.rnn.pad_sequence([b["mel"] for b in batch], batch_first=True)
    lengths = torch.tensor([b["mel"].size(0) for b in batch], dtype=torch.long)
    labels = torch.cat([b["labels"] for b in batch])
    label_lengths = torch.tensor([b["labels"].numel() for b in batch], dtype=torch.long)
    # Cluster targets are optional. When present, they're padded with -100
    # (the standard CE ignore_index) so the loss naturally masks padded
    # positions. We return them as None if the dataset wasn't built with
    # ssl_targets_path — caller checks.
    if "z100" in batch[0]:
        z100 = nn.utils.rnn.pad_sequence(
            [b["z100"] for b in batch], batch_first=True, padding_value=-100,
        )
        z500 = nn.utils.rnn.pad_sequence(
            [b["z500"] for b in batch], batch_first=True, padding_value=-100,
        )
        target_lengths = torch.tensor([b["z100"].numel() for b in batch], dtype=torch.long)
    else:
        z100 = z500 = target_lengths = None
    return mel, lengths, labels, label_lengths, z100, z500, target_lengths


def collate_eval(batch):
    mel, lengths, labels, label_lengths, _z100, _z500, _tlens = collate_ctc(batch)
    transcripts = [b["transcript"] for b in batch]
    return mel, lengths, labels, label_lengths, transcripts


class DistillDataset(Dataset):
    """Unlabeled dataset for DistilHuBERT-style feature distillation.

    Yields, per utterance:
      - ``mel``: CMVN log-mel [T, 80] for the student (uses LogMelExtractor,
        including its pre-emphasis + CMVN — identical to the SSL front-end).
      - ``waveform``: raw mono waveform [T_wav] at 16 kHz for the frozen
        teacher. Loaded WITHOUT pre-emphasis or per-utterance normalization,
        matching what the released HuBERT/wav2vec2 Base checkpoints expect.

    No labels are used — distillation operates on the unlabeled 960h corpus,
    mirroring the SSL pretraining stage it replaces.
    """

    def __init__(
        self,
        data_root,
        splits,
        cmvn_path,
        mel_cache_dir=None,
        max_items=None,
        teacher_sample_rate=16000,
        max_duration_sec=None,
        compute_durations=False,
        durations_cache=None,
    ):
        self.items = list(iter_librispeech_items(data_root, splits))
        if max_items is not None:
            self.items = self.items[:max_items]
        self.extractor = LogMelExtractor()
        self.mean, self.std = load_cmvn(cmvn_path)
        self.mel_cache_dir = mel_cache_dir
        self.teacher_sample_rate = teacher_sample_rate
        if mel_cache_dir:
            self.items = [
                item for item in self.items
                if os.path.isfile(os.path.join(mel_cache_dir, item["split"], item["uid"] + ".pt"))
            ]
        # Optional duration handling. Needed to (a) drop over-long utterances
        # that would blow up the frozen teacher's activation memory, and/or
        # (b) length-bucket batches to cut padding waste. Only scans durations
        # when actually requested — keeps startup fast in the default path.
        self.durations = None
        if max_duration_sec is not None or compute_durations:
            durs = self._compute_durations(self.items, durations_cache)
            kept_items, kept_durs = [], []
            for item, dur in zip(self.items, durs):
                if dur is None:
                    continue  # unreadable header — skip rather than crash mid-run
                if max_duration_sec is not None and dur > max_duration_sec:
                    continue
                kept_items.append(item)
                kept_durs.append(dur)
            self.items = kept_items
            self.durations = kept_durs

    @staticmethod
    def _compute_durations(items, cache_path=None):
        """Return per-item audio durations (seconds), using a JSON uid->sec cache.

        The cache makes repeat runs cheap: the first run computes any missing
        durations (torchaudio.info is header-only, ~ms/file) and writes the
        merged cache back atomically. Concurrent ranks may race the write, but
        os.replace is atomic and the content is identical, so last-writer-wins
        is harmless.
        """
        cache = {}
        if cache_path and os.path.isfile(cache_path):
            try:
                with open(cache_path, encoding="utf-8") as f:
                    cache = json.load(f)
            except Exception:
                cache = {}
        durations = []
        computed_any = False
        for item in items:
            dur = cache.get(item["uid"])
            if dur is None:
                try:
                    dur = audio_duration_seconds(item["audio_path"])
                except Exception:
                    dur = None
                computed_any = True
            durations.append(dur)
        if cache_path and computed_any:
            merged = dict(cache)
            for item, dur in zip(items, durations):
                if dur is not None:
                    merged[item["uid"]] = dur
            try:
                os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
                tmp = f"{cache_path}.tmp.{os.getpid()}"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(merged, f)
                os.replace(tmp, cache_path)
            except Exception:
                pass
        return durations

    def __len__(self):
        return len(self.items)

    def _load_raw_waveform(self, path):
        try:
            wav, sr = torchaudio.load(path, normalize=True)
        except Exception:
            import soundfile as sf
            wav_np, sr = sf.read(path, dtype="float32", always_2d=True)
            wav = torch.from_numpy(wav_np.T)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.teacher_sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.teacher_sample_rate)
        return wav.squeeze(0)  # [T_wav]

    def __getitem__(self, idx):
        item = self.items[idx]
        if self.mel_cache_dir:
            mel = torch.load(
                os.path.join(self.mel_cache_dir, item["split"], item["uid"] + ".pt"),
                map_location="cpu",
            ).float()
        else:
            mel = apply_cmvn(self.extractor(item["audio_path"]), self.mean, self.std)
        waveform = self._load_raw_waveform(item["audio_path"])
        return {"uid": item["uid"], "mel": mel, "waveform": waveform}


def collate_distill(batch):
    """Collate (mel, waveform) pairs for distillation.

    Returns:
        mel:         [B, T_mel, 80] padded student input
        lengths:     [B] valid mel frame counts
        waveforms:   [B, T_wav] padded teacher input
        wav_lengths: [B] valid sample counts
    """
    mel = nn.utils.rnn.pad_sequence([b["mel"] for b in batch], batch_first=True)
    lengths = torch.tensor([b["mel"].size(0) for b in batch], dtype=torch.long)
    wavs = [b["waveform"] for b in batch]
    wav_lengths = torch.tensor([w.size(0) for w in wavs], dtype=torch.long)
    waveforms = nn.utils.rnn.pad_sequence(wavs, batch_first=True)
    return mel, lengths, waveforms, wav_lengths


class LengthBucketSampler(torch.utils.data.Sampler):
    """Groups items of similar length into buckets, shuffles within/across
    buckets, and shards across DDP ranks.

    Reduces padding waste (and thus peak memory) on variable-length audio: a
    batch of similarly-long utterances pads far less than a random batch that
    mixes a 2 s clip with a 30 s clip. ``lengths`` may be any per-item size
    proxy (audio seconds or frame counts) — only the ordering matters.
    """

    def __init__(self, lengths, batch_size, num_replicas=1, rank=0,
                 bucket_size_multiplier=100, seed=0, epoch=0):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.num_replicas = max(1, num_replicas)
        self.rank = rank
        self.bucket_size = batch_size * bucket_size_multiplier
        self.seed = seed
        self.epoch = epoch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.argsort(torch.tensor(self.lengths, dtype=torch.float)).tolist()
        buckets = [indices[i:i + self.bucket_size] for i in range(0, len(indices), self.bucket_size)]
        for bucket in buckets:
            perm = torch.randperm(len(bucket), generator=g).tolist()
            bucket[:] = [bucket[p] for p in perm]
        bucket_order = torch.randperm(len(buckets), generator=g).tolist()
        flat = [idx for b in bucket_order for idx in buckets[b]]
        total = (len(flat) // self.num_replicas) * self.num_replicas
        flat = flat[self.rank:total:self.num_replicas]
        return iter(flat)

    def __len__(self):
        return len(self.lengths) // self.num_replicas


class BatchSpecAugment(nn.Module):
    """SpecAugment for padded [B, T, F] CMVN log-mel batches.

    mask_value can be a scalar (e.g. 0.0) or a 1-D tensor of size n_mels
    (e.g. the SSL-trained ``mask_emb``) — in the tensor case, the same
    learnable per-mel mask vector replaces every masked spectrogram cell,
    matching the distribution the encoder saw during SSL pretraining."""

    def __init__(
        self,
        time_mask_param=40,
        freq_mask_param=30,
        num_time_masks=2,
        num_freq_masks=2,
        mask_value=0.0,
    ):
        super().__init__()
        self.time_mask_param = int(time_mask_param)
        self.freq_mask_param = int(freq_mask_param)
        self.num_time_masks = int(num_time_masks)
        self.num_freq_masks = int(num_freq_masks)
        if isinstance(mask_value, torch.Tensor):
            if mask_value.dim() != 1:
                raise ValueError("SpecAugment tensor mask_value must be 1-D [n_mels].")
            self.register_buffer("mask_value_tensor", mask_value.float())
            self.mask_value = 0.0
        else:
            self.register_buffer("mask_value_tensor", None)
            self.mask_value = float(mask_value)

    def _fill(self, out, index, live_value):
        # live_value (when provided) takes precedence over the registered buffer:
        # used during SSL pretraining so SpecAug fills with the *current* mask_emb
        # parameter and gradients can flow back into mask_emb. The buffer path is
        # used at fine-tune time where mask_emb is loaded once and frozen.
        source = live_value if live_value is not None else self.mask_value_tensor
        if source is None:
            out[index] = self.mask_value
        else:
            value = source.to(dtype=out.dtype, device=out.device)
            if isinstance(index, tuple) and len(index) == 3:
                value = value[index[2]]
            out[index] = value

    def forward(self, mel, lengths, mask_value=None):
        if self.num_time_masks <= 0 and self.num_freq_masks <= 0:
            return mel
        out = mel.clone()
        batch, _, n_mels = out.shape
        device = out.device
        live = mask_value if (isinstance(mask_value, torch.Tensor) and mask_value.dim() == 1) else None

        for b in range(batch):
            valid_t = int(lengths[b].item())
            if valid_t <= 0:
                continue
            for _ in range(self.num_freq_masks):
                width_max = min(self.freq_mask_param, n_mels)
                if width_max <= 0:
                    continue
                width = int(torch.randint(0, width_max + 1, (1,), device=device).item())
                if width == 0 or width >= n_mels:
                    continue
                start = int(torch.randint(0, n_mels - width + 1, (1,), device=device).item())
                self._fill(out, (b, slice(0, valid_t), slice(start, start + width)), live)
            for _ in range(self.num_time_masks):
                width_max = min(self.time_mask_param, valid_t)
                if width_max <= 0:
                    continue
                width = int(torch.randint(0, width_max + 1, (1,), device=device).item())
                if width == 0 or width >= valid_t:
                    continue
                start = int(torch.randint(0, valid_t - width + 1, (1,), device=device).item())
                self._fill(out, (b, slice(start, start + width), slice(None)), live)
        return out
