# LM_Dec_AttnFree_Mamba

Mamba - AttnFree family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Benchmarked**
- Failure class: none

## Provenance

- Source repository: LM_2023_Mamba_transformer-pytorch
- Upstream / license: pscan from alxndrTL/mamba.py (MIT); Mamba blocks: source not stated in repo - confirm

## Differences from the source ([Unified] protocol)

- LM head (embedding, final RMSNorm, linear) written here; Lightning MambaLLM not used; d_model 1024, d_state 16, parallel scan
- micro-batch 8 with gradient accumulation (same update): the pure-PyTorch scan keeps about 0.9 MB per token, so batch 32 x 512 exceeds 8 GB; on Windows this spilled into system RAM and the first LM-05 run reached only 500 steps in 2 h (kept as lm05_capped_1019.json)
- BUG FIX in vendored pscan (2026-10-19): backward returned gradients in the wrong layout and used the already-scanned A, so gradients were wrong; also padded to power-of-two lengths. Verified equal to the sequential scan (forward and gradients, error < 1e-14)

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 10.82 M
- Causality check: pass (largest change of earlier outputs 0.0e+00)
- Overfit check (4 x 256 bytes): 5.715 -> 0.048 nats, pass
- LM-02 enwik8 (200M tokens): valid 1.604 / test 1.610 bits per byte
- Peak GPU memory at length 2048 (batch 1): train 1800 MB, inference 469 MB
- Peak GPU memory at length 8192 (batch 1): train 7256 MB, inference 1820 MB
- LM-05 recall `lm05v3`: 4 pairs @ 512: 0.01 (keys removed 0.01); 4 pairs @ 2048: 0.01 (keys removed 0.02); 16 pairs @ 512: 0.02 (keys removed 0.02); 16 pairs @ 2048: 0.02 (keys removed 0.01)

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy
python run.py check mamba
python run.py lm02 mamba      # enwik8 is downloaded to data/ on first use
```
