# Pseudocode

## Target Generation

```python
def compute_targets(utterances, chunk_size=4, chunk_stride=2, pca_dim=64):
    # utterances: list of (utt_id, waveform)

    mels = {}
    for utt_id, wav in utterances:
        mel = log_mel(wav, n_mels=80, win_ms=25, hop_ms=10)  # [T, 80]
        mels[utt_id] = mel

    mean, std = global_cmvn_stats(mels.values())             # [80], [80]

    chunks = []
    index = []
    for utt_id, mel in mels.items():
        mel = (mel - mean) / std
        for start in range(0, len(mel) - chunk_size + 1, chunk_stride):
            chunk = mel[start:start + chunk_size]            # [C, 80]
            chunks.append(chunk.reshape(-1))                 # [C * 80]
            index.append((utt_id, start))

    train_sample = sample(chunks, max_items=5_000_000)

    pca = PCA(n_components=pca_dim, whiten=True)
    pca.fit(train_sample)

    reduced_sample = pca.transform(train_sample)

    km100 = KMeans(n_clusters=100, init="k-means++", max_iter=300)
    km500 = KMeans(n_clusters=500, init="k-means++", max_iter=300)
    km100.fit(reduced_sample)
    km500.fit(reduced_sample)

    targets = defaultdict(list)
    for flat, (utt_id, start) in zip(chunks, index):
        y = pca.transform(flat[None, :])
        z100 = int(km100.predict(y)[0])
        z500 = int(km500.predict(y)[0])
        targets[utt_id].append({
            "start": start,
            "z100": z100,
            "z500": z500,
        })

    save_artifacts(mean, std, pca, km100, km500, targets)
```

## Causal SSL Pretraining

```python
def pretrain_ssl(model, loader, optimizer, delay=2):
    model.train()

    for batch in loader:
        mel = batch["mel"]          # [B, T, 80]
        z100 = batch["z100"]        # [B, T']
        z500 = batch["z500"]        # [B, T']
        lengths = batch["lengths"]

        h = model.encoder(mel)      # [B, T', D], causal

        pred100 = model.head100(h[:, :-delay])   # [B, T'-d, 100]
        pred500 = model.head500(h[:, :-delay])   # [B, T'-d, 500]

        target100 = z100[:, delay:]              # [B, T'-d]
        target500 = z500[:, delay:]              # [B, T'-d]

        valid_mask = make_valid_mask(lengths, pred100.shape[1])

        loss100 = masked_cross_entropy(pred100, target100, valid_mask)
        loss500 = masked_cross_entropy(pred500, target500, valid_mask)
        loss = loss100 + loss500

        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

## CTC Fine-Tuning

```python
def finetune_ctc(ssl_checkpoint, model, loader, optimizer, tokenizer):
    model.encoder.load_state_dict(ssl_checkpoint["encoder"])
    model.replace_ssl_heads_with_ctc_head(vocab_size=tokenizer.vocab_size)

    for batch in loader:
        mel = batch["mel"]                    # [B, T, 80]
        labels = batch["labels"]
        label_lengths = batch["label_lengths"]
        input_lengths = batch["input_lengths"]

        h = model.encoder(mel)                # [B, T', D]
        logits = model.ctc_head(h)            # [B, T', vocab]
        log_probs = log_softmax(logits, dim=-1)

        output_lengths = model.output_lengths(input_lengths)
        loss = ctc_loss(
            log_probs.transpose(0, 1),
            labels,
            output_lengths,
            label_lengths,
        )

        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

