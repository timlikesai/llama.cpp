# DSpark speculative decoding drafters for Inkling-2-Mini

DSpark drafter modules converted from Inkling 2 Mini's own 3-layer block drafter (`mtp.safetensors`, block size 8),
for llama.cpp speculative decoding.

These are **optional add-ons**. The quants in this repo work exactly the same without them, and neither file is
loaded unless you ask for it with `--spec-type draft-dspark`.

## Files

| File | Location | Size | Projections (attention, MLP) | Markov head | Norms and biases |
|---|---|---|---|---|---|
| `dspark-Inkling-2-Mini-Q8_0.gguf` | repo root | 0.40 GB | Q8_0 | BF16 | F32 |
| `dspark-Inkling-2-Mini-BF16.gguf` | `dspark/` | 0.65 GB | BF16, bit exact | BF16 | F32 |

Both files have 54 tensors and `general.architecture = dflash`. The source checkpoint is BF16 throughout. They differ
only in the 27 attention and MLP projections: `BF16` keeps them byte for byte, `Q8_0` stores them as Q8_0 (worst
relative RMSE 0.62%). The Markov head (`markov_w1`, `dspark_hidden_out`, `joint_proj`) stays BF16 and every F32 tensor
(norms, biases, the mask embedding and the 4x4 feature fusion) is the exact F32 upcast of the BF16 source, byte
identical in both files. The token embeddings and the LM head are not in the files: the drafter borrows the target's.

**Which one:** `Q8_0` is the default. It sits at the repo root, so llama.cpp and Unsloth Studio find it automatically
for every quant in this repo. It accepts drafts at the same rate as `BF16` in our tests. Take `BF16` only if you want
an exact reproduction of the original weights.

## llama.cpp version

Inkling 2 Mini runs its middle layers several times per token, and its DSpark drafter taps those looped layers. Use
the Unsloth llama.cpp prebuilt (unslothai/llama.cpp releases), which carries looped-layer support and the DSpark
drafter as pins.

## Usage

```bash
llama-server -hf unsloth/Inkling-2-Mini-GGUF:UD-Q4_K_XL \
  --spec-type draft-dspark \
  --spec-draft-n-max 3
```

With local files, or for the `BF16` drafter, pass the drafter path with `-md`:

```bash
llama-server \
  -m Inkling-2-Mini-UD-Q4_K_XL.gguf \
  -md dspark/dspark-Inkling-2-Mini-BF16.gguf \
  --spec-type draft-dspark --spec-draft-n-max 3 -ngld 99
```

- `--spec-draft-n-max` is clamped to 7: the trained block size is 8 (`block_size` in `mtp_config` of the original
  `config.json`), and the first slot of each block is the last accepted token.
- Do not pin the drafter to one device (`-devd`): it borrows the target's embeddings and output head, so it has to span
  the same devices as the target.
- `--mtp` / `--spec-type draft-mtp` do not apply: Inkling 2 Mini ships a DSpark block drafter, not an MTP head.

## Notes

`--spec-draft-n-max 3` is llama.cpp's default and what Unsloth Studio uses. Values up to 7 are accepted: longer
drafts land more tokens per step but spend more draft compute on tokens the target rejects.

As with other speculative drafters in llama.cpp, greedy output with the drafter is not guaranteed to be bit identical
to a run without it: the target's batched verification can flip near-tie tokens on CUDA. It is not a drafter defect.
