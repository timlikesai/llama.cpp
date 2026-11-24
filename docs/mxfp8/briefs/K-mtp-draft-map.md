# Brief K — Map Inkling-2's MTP head (`mtp.safetensors`) onto llama.cpp's `dflash`/`dspark` draft format

**Deliverable:** `docs/mxfp8/12-mtp-draft-plan.md` — an implementation-ready tensor-by-tensor mapping + the exact metadata a `dflash` draft GGUF needs, so a later task writes the converter without further research.
**Read-only** except that one file. Do not edit `conversion/` (another agent is in there right now), do not build, no docker/GPU commands.
Write the file within your first 5 tool calls with headings + raw bullets, then refine. **20-minute budget.** Cite `file:line`. Mark anything unproven `UNVERIFIED`.

## Goal

The checkpoint ships a 3-layer MTP drafter that master's converter currently discards (`conversion/inkling2.py:190` drops any name containing `mtp.`, `markov_head`, `nextn_predict`). We want to generate it as a draft GGUF loadable with `--model-draft`, matching what the ecosystem already ships and verifies against.

## Ground truth available locally (use it — this is a differential problem, not a design problem)

- **Input:** `~/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/1fb4134b9a74f9681054f4ef82e4bf1f458a2337/mtp.safetensors` — 635,801,416 B, **54 BF16 tensors**: `fuse_weight (4,4)`, and per layer `0..2`: `dflare_norm.weight`, `input_layernorm.weight`, `post_attention_layernorm.weight`, `self_attn.{q,k,v,o}_proj.weight` (+ `k_proj_ctx`, `v_proj_ctx`), `self_attn.{q,k}_norm.weight`, `mlp.{gate,up,down}_proj.weight`. Shapes: q_proj `(3072,2304)`, k/v_proj `(1024,2304)`, k/v_proj_ctx `(1024,2304)`, o_proj `(2304,3072)`, mlp gate/up `(9216,2304)`, down `(2304,9216)`.
  `config.json → mtp_config`: hidden 2304, 3 layers, 24 heads, 8 kv heads, head_dim 128, intermediate 9216, `rope_theta 1e6`, `local_layer_ids [0,2]`, `sliding_window_size 1024`, `block_size 8`, `markov_rank 256`, `markov_head_type "rnn_hidden"`, `target_layer_ids [5,23,41,48]`.
- **Reference output:** `~/models/unsloth/Inkling-2-Mini-GGUF/dspark-Inkling-2-Mini-Q8_0.gguf` — 402,278,560 B, `general.architecture = dflash`, `general.file_type = 7`, **54 tensors** (24 F32, 27 Q8_0, 3 BF16), names like `dspark_fuse.weight`, `dspark_hidden_out.{weight,bias}`, `dspark_norm.weight`, `joint_proj.{weight,bias}`. Read it with `sys.path.insert(0,'gguf-py'); from gguf import GGUFReader` (ReaderTensor exposes `.name/.shape/.tensor_type/.data/.n_bytes`; `.data` is a memmap in **element** dims; call `.contents()` on a ReaderField). It is 0.4 GB so dumping all of it is fine.
- Master runtime side: `src/models/dflash.cpp`, `src/models/dspark.cpp`, `LLM_TENSOR_NEXTN*` (22) and `LLM_KV_*` in `src/llama-arch.cpp`, draft plumbing in `src/llama-model.cpp` (`layer.nextn.*`), plus this box already runs a DFlash2 drafter for Qwen3.8 — check `~/code/services/model-presets/` and `~/code/services/docker-compose.yml` for the flags it uses (`--spec-draft-n-max 7` etc.).
- Converter side precedents to imitate: find which `conversion/*.py` produce `dflash`/`dspark`/nextn drafts (grep `conversion/` for `dflash`, `dspark`, `nextn`, `mtp`) and how a second output file is written (does `convert_hf_to_gguf.py` support emitting a draft at all, or is it a separate invocation / separate repo dir? — determine this and state it plainly, it decides the UX).

## Required content

1. **Complete tensor table**: HF name → dspark GGUF name, dtype change, shape (with GGUF's reversed dim order), and per-tensor evidence from the reference file. Flag any HF tensor with **no** counterpart (e.g. the markov/rnn-hidden head, per-layer norms folded elsewhere?) and any reference tensor with no HF source (e.g. `dspark_hidden_out.bias`, `joint_proj.bias` — synthesized zeros? derived? say which, by comparing values if a counterpart exists).
2. **Quantization mapping**: which of the 54 landed as Q8_0 vs F32 vs BF16 and the rule (weights → Q8_0? norms/e_proj → F32?). Note the reference is Q8_0 — for a *lossless* project we may want BF16 pass-through; state the byte cost of each option (635 MB BF16 vs 402 MB Q8_0).
3. **Metadata checklist** the draft GGUF must carry (`general.architecture=dflash`, block_count, embedding/ffn/head counts, `*_nextn.*` keys, rope theta, swa window/pattern for `local_layer_ids [0,2]`, context length, vocab + tokenizer keys needed for a draft, and anything `dflash.cpp`/`dspark.cpp` reads via `load_arch_hparams` that would make load fail if absent) — derive it by reading the reference file's full KV dump **and** `dflash.cpp`/`dspark.cpp` hparams loading.
4. **How it is consumed**: exact `llama-server`/`llama-cli` flags for target+draft (from master's code/docs, and the Qwen3.8 example on this box), and what `n_layer_nextn`/`target_layer_ids` imply for the loop model's 48-logical-layer indexing.
5. **Implementation plan**: files to touch (a draft-export hook in the Inkling conversion module vs a separate `--draft-out` path), the 3 riskiest mismatches, and a verification recipe (byte/value comparison of our produced draft vs the reference dspark file, plus `--model-draft` acceptance-test with an actual draft-load on CPU).
6. **`## Corrections to my brief`** if anything above is wrong.
