# Loss Choice: InfoNCE vs MSE vs Cross-Entropy

## Short Answer

For the proposed discrete k-means target setup, the best first loss is:

```text
cross-entropy over k-means unit IDs
```

Use:

```text
L = CE(z^100) + CE(z^500)
```

Between InfoNCE and MSE:

```text
InfoNCE is better than MSE if the goal is learning predictive discrete acoustic
representations.
```

But for this exact plan, cross-entropy is simpler and cleaner than both.

## Why Cross-Entropy First

The targets are discrete cluster IDs:

```text
z_t^100 in {1, ..., 100}
z_t^500 in {1, ..., 500}
```

So the natural objective is classification:

```text
predict cluster ID -> cross entropy
```

Advantages:

```text
simple
stable
easy to debug
no negative sampling
matches the k-means target format
works well for HuBERT-like discrete-unit prediction
```

Recommended:

```text
L = CE(head100(h_t), z_{t+d}^100) + CE(head500(h_t), z_{t+d}^500)
```

## InfoNCE

InfoNCE predicts the correct future representation among negative samples.

Generic form:

```text
L_InfoNCE = -log exp(sim(c_t, y_{t+d}) / tau)
                 / sum_j exp(sim(c_t, y_j) / tau)
```

Pros:

```text
strong predictive representation learning objective
good for CPC-style causal SSL
does not require discrete labels if using continuous targets
encourages discriminative future prediction
```

Cons:

```text
requires negative sampling
more sensitive to batch size and temperature
more implementation complexity
false negatives can hurt
harder to explain/debug than CE over k-means IDs
```

Use InfoNCE if:

```text
you predict PCA chunk embeddings directly
or you want a CPC-style baseline
or CE over cluster IDs underperforms
```

For this project, InfoNCE is a good ablation, not the first main objective.

## MSE

MSE predicts continuous targets directly.

Example:

```text
L_MSE = || prediction_t - PCA(chunk_{t+d}) ||^2
```

Pros:

```text
very simple
no clusters needed
stable numerically
easy reconstruction-style baseline
```

Cons:

```text
often learns low-level spectrogram texture
can average over ambiguous futures
less discriminative than CE or InfoNCE
may transfer worse to ASR
```

Use MSE for:

```text
masked reconstruction baseline
continuous PCA future-prediction baseline
```

Do not use MSE as the main method if the paper claim is about discrete
spectrogram units.

## Recommended Loss Ranking

For the first paper version:

```text
1. Cross-entropy over k-means units
2. InfoNCE over PCA chunk embeddings
3. MSE over PCA chunk embeddings or spectrogram frames
```

Expected ASR usefulness:

```text
CE discrete units > InfoNCE continuous targets > MSE reconstruction
```

This is a hypothesis, so it should be tested.

## Minimal Ablation

Run these three:

```text
scratch ASR
MSE SSL -> CTC fine-tune
CE k-means SSL -> CTC fine-tune
```

If compute allows:

```text
InfoNCE SSL -> CTC fine-tune
```

This directly answers whether discrete prediction is better than continuous
minimization/reconstruction.

