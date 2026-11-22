from __future__ import annotations

import json
import re
from pathlib import Path
from functools import partial
from typing import Callable, Iterable, TYPE_CHECKING, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, ModelBase, MmprojModel, TextModel, gguf, logger


def _e4m3_table() -> np.ndarray:
    codes = np.arange(256, dtype=np.uint8)
    exponent = ((codes & 0x7F) >> 3).astype(np.int32)
    mantissa = (codes & 7).astype(np.float32)
    values = np.where(exponent == 0, np.ldexp(mantissa, -9), np.ldexp(1 + mantissa / 8, exponent - 7))
    values = np.where(codes & 0x80, -values, values)
    values[(codes & 0x7F) == 0x7F] = np.nan
    return values.astype(np.float32)


_E4M3 = _e4m3_table()
_E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
_EXPERT_CHUNK_ROWS = 128
_QK_MXFP8 = 32  # ggml/src/ggml-common.h: QK_MXFP8, one ue8m0 scale per 32 e4m3 codes
_QK_NVFP4 = 64  # ggml/src/ggml-common.h: QK_NVFP4, one e4m3 scale per 16 elements, 4 per 64-element block
_QK_NVFP4_SUB = 16


def _dequant_expert_chunk(weight: Callable[[], Tensor], scale: Callable[[], Tensor], amax: Callable[[], Tensor] | None, denominator: Callable[[], Tensor] | None, name: str, expert: int, rows: slice) -> np.ndarray:
    # Fresh lazy tensors release their mappings after each chunk instead of caching the layer.
    packed = LazyTorchTensor.to_eager(weight().view(torch.uint8)[expert, rows]).numpy()
    scales = LazyTorchTensor.to_eager(scale().view(torch.uint8)[expert, rows]).numpy()
    if amax is not None:
        block_scale = _E4M3[scales]
        if np.isnan(block_scale).any():
            raise ValueError(f"Inkling2: {name}_scale expert {expert}: E4M3FN NaN")
        values = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.float32)
        values[:, 0::2] = _E2M1[packed & 15]
        values[:, 1::2] = _E2M1[packed >> 4]
        values.reshape(values.shape[0], -1, 16)[:] *= block_scale[:, :, None]
        assert denominator is not None
        a = LazyTorchTensor.to_eager(amax()[expert]).item()
        d = LazyTorchTensor.to_eager(denominator()).item()
        if not np.isfinite(a) or a < 0:
            raise ValueError(f"Inkling2: {name}_amax expert {expert}: expected finite nonnegative value")
        if not np.isfinite(d) or d <= 0:
            raise ValueError(f"Inkling2: {name}_outer_scale_denominator expert {expert}: expected finite positive value")
        with np.errstate(over="ignore", invalid="ignore"):
            values *= np.float32(a) / np.float32(d)
    else:
        if np.any(scales == 255):
            raise ValueError(f"Inkling2: {name}_scale expert {expert}: UE8M0 byte 255 is NaN")
        values = _E4M3[packed]
        if np.isnan(values).any():
            raise ValueError(f"Inkling2: {name} expert {expert}: E4M3FN NaN")
        block_scale = np.ldexp(np.ones(scales.shape, dtype=np.float32), scales.astype(np.int32) - 127)
        with np.errstate(over="ignore", invalid="ignore"):
            values.reshape(values.shape[0], -1, 32)[:] *= block_scale[:, :, None]
    if not np.isfinite(values).all():
        raise ValueError(f"Inkling2: {name} expert {expert}: non-finite dequantized weight")
    return values


def _expert_row_slices(experts: int, rows: int, w13: bool) -> Iterable[tuple[int, slice]]:
    # Row chunks of a stacked expert tensor, in GGUF row order. A fused w13 interleaves
    # gate and up rows in the checkpoint but is stored as all gate rows then all up rows,
    # so both chunks and the slice step depend on w13. One slice selects payload rows and
    # their scale rows, so the two can never be permuted differently.
    for expert in range(experts):
        for half in range(2 if w13 else 1):
            for start in range(0, rows // 2 if w13 else rows, _EXPERT_CHUNK_ROWS):
                stop = min(start + _EXPERT_CHUNK_ROWS, rows // 2 if w13 else rows)
                yield expert, (slice(2 * start + half, 2 * stop + half, 2) if w13 else slice(start, stop))


def _pack_nvfp4_chunk(weight: Callable[[], Tensor], scale: Callable[[], Tensor], name: str, expert: int, rows: slice) -> np.ndarray:
    # Lossless apart from the nibble order: the checkpoint pairs element 2i with 2i+1 in one byte, ggml
    # pairs j with j + 8 inside each 16-element sub-block (conversion/base.py:1320-1325), and a super-block
    # is the 4 sub-block scales followed by their 32 payload bytes.
    packed = LazyTorchTensor.to_eager(weight().view(torch.uint8)[expert, rows]).numpy()
    scales = LazyTorchTensor.to_eager(scale().view(torch.uint8)[expert, rows]).numpy()
    # ggml stores the sub-block scales as UE4M3, so a negative scale cannot be kept without requantizing
    if np.any(scales & 0x80):
        raise ValueError(f"Inkling2: {name}_scale expert {expert}: NVFP4 scale sign bit set, cannot keep as UE4M3")
    if np.isnan(_E4M3[scales]).any():
        raise ValueError(f"Inkling2: {name}_scale expert {expert}: E4M3FN NaN")
    if packed.shape[-1] * 2 != scales.shape[-1] * _QK_NVFP4_SUB:
        raise ValueError(f"Inkling2: {name} expert {expert}: payload {packed.shape} and scale {scales.shape} disagree")
    n_sub = _QK_NVFP4 // _QK_NVFP4_SUB
    if scales.shape[-1] % n_sub:
        raise ValueError(f"Inkling2: {name} expert {expert}: {scales.shape[-1]} sub-blocks is not a multiple of {n_sub}")
    codes = np.stack((packed & 15, packed >> 4), axis=-1).reshape(packed.shape[0], -1, n_sub, _QK_NVFP4_SUB)
    raw = np.empty((packed.shape[0], codes.shape[1], _QK_NVFP4 // 2 + n_sub), dtype=np.uint8)
    raw[:, :, :n_sub] = scales.reshape(packed.shape[0], codes.shape[1], n_sub)
    raw[:, :, n_sub:] = (codes[:, :, :, : _QK_NVFP4_SUB // 2] | (codes[:, :, :, _QK_NVFP4_SUB // 2:] << 4)).reshape(packed.shape[0], codes.shape[1], _QK_NVFP4 // 2)
    return raw.reshape(packed.shape[0], -1)


def _pack_mxfp8_chunk(weight: Callable[[], Tensor], scale: Callable[[], Tensor], name: str, expert: int, rows: slice) -> np.ndarray:
    # Lossless: the e4m3 codes and the ue8m0 scale bytes are copied verbatim into the
    # ggml block_mxfp8 order, one scale byte per 32 code bytes. Only bytes move, no value is recomputed.
    packed = LazyTorchTensor.to_eager(weight().view(torch.uint8)[expert, rows]).numpy()
    scales = LazyTorchTensor.to_eager(scale().view(torch.uint8)[expert, rows]).numpy()
    if np.any(scales == 255):
        raise ValueError(f"Inkling2: {name}_scale expert {expert}: UE8M0 byte 255 is NaN")
    values = _E4M3[packed]
    if np.isnan(values).any():
        raise ValueError(f"Inkling2: {name} expert {expert}: E4M3FN NaN")
    blocks = packed.reshape(packed.shape[0], -1, _QK_MXFP8)
    raw = np.empty(blocks.shape[:2] + (_QK_MXFP8 + 1,), dtype=np.uint8)
    raw[:, :, 0] = scales
    raw[:, :, 1:] = blocks
    return raw.reshape(packed.shape[0], -1)


def _is_inkling2(dir_model: Path) -> bool:
    if not (dir_model / "config.json").is_file():
        return False
    with open(dir_model / "config.json", encoding="utf-8") as f:
        return json.load(f).get("model_type") in ("inkling2", "inkling2_text", "inkling2_loop", "inkling2_loop_text")


@ModelBase.register_hparams_loader(_is_inkling2)
def _load_inkling2_hparams(dir_model: Path):
    # AutoConfig injects MXFP8 defaults and can replace the original layer schedule.
    with open(dir_model / "config.json", encoding="utf-8") as f:
        return json.load(f)


@ModelBase.register("Inkling2ForConditionalGeneration", "Inkling2ForCausalLM")
@ModelBase.example("thinkingmachines/Inkling-2-Small")
class Inkling2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.INKLING2

    # The checkpoint's mxfp8_nearest and nvfp4_nearest experts are already ggml block values, so they
    # can be repacked bit-exactly instead of dequantized. Opt-in: convert_hf_to_gguf.py sets
    # native_experts from --native-experts.
    supports_native_experts = True
    native_experts = False
    supports_mtp_export = True  # the drafter is embedded as nextn layers; --no-mtp drops it deliberately

    # the drafter is indexed in a second pass over mtp.safetensors; the flag selects its name mapping
    _indexing_draft = False
    _draft_beg = 0     # trunk layer count: embedded nextn layers keep absolute block ids from here
    _draft_count = 0

    def __init__(self, dir_model: Path, *args, **kwargs):
        self._expert_quant: dict[str, dict] = {}
        self._packed_experts: dict[str, gguf.LazyChunkedTensor] = {}
        quant_path = dir_model / "hf_quant_config.json"
        if quant_path.is_file():
            with open(quant_path, encoding="utf-8") as f:
                groups = json.load(f).get("model", {}).get("parameter_groups", [])
            for group in groups:
                for name in group.get("tensors", []):
                    item = self.filter_tensors((name, lambda: torch.empty(0)))
                    if item is not None and item[0].endswith((".w13_weight", ".w2_weight")):
                        name = item[0]
                        if name in self._expert_quant:
                            raise ValueError(f"Inkling2: duplicate hf_quant_config.json scheme for {name}")
                        self._expert_quant[name] = group
        # the drafter ships inside the checkpoint: mtp.safetensors holds 54 BF16 tensors - 3 draft
        # stages, the DFlare fuse matrix and the rnn_hidden markov head. Embed it as nextn layers, the
        # same mechanism the converters that keep MTP in the main file use (deepseek, deepseek41,
        # bailingmoe, command_r), so the GGUF carries every weight the model was released with.
        with open(dir_model / "config.json", encoding="utf-8") as f:
            cfg = json.load(f)
        self.mtp: dict = cfg.get("mtp_config", {})
        # convert_hf_to_gguf sets no_mtp as a class attribute before constructing, so read it either way
        no_mtp = bool(kwargs.get("no_mtp", getattr(type(self), "no_mtp", False)))
        self.n_draft = 0 if no_mtp else int(self.mtp.get("num_hidden_layers", 0))
        type(self)._draft_beg = int(cfg["text_config"]["num_hidden_layers"])
        type(self)._draft_count = self.n_draft
        if self.n_draft and not (dir_model / "mtp.safetensors").is_file():
            raise ValueError(f"Inkling2: mtp_config declares {self.n_draft} draft layers but mtp.safetensors is missing")
        super().__init__(dir_model, *args, **kwargs)
        logger.warning("Inkling2: converting the text backbone with%s; vision tensors go to the mmproj"
                       % (f" the {self.n_draft}-layer drafter embedded as nextn layers" if self.n_draft else " no drafter"))
        if self.native_experts:
            logger.warning("Inkling2: --native-experts keeps the MXFP8 and NVFP4 expert layers as native bytes")
        hp = self.hparams
        for name, default in (("q_bias", False), ("o_bias", False), ("n_shared_experts", 0), ("use_value_norm_no_gamma", True), ("attn_res_normalize_values", True), ("scoring_func", "sigmoid")):
            if hp.get(name, default) != default:
                raise ValueError(f"Inkling2: unsupported {name}={hp[name]!r}")
        types = hp.get("layer_types")
        if types is None:
            local = hp["local_layer_ids"]
            types = ["hybrid_sliding" if i in local else "hybrid" for i in range(self.block_count)]
        if len(types) != self.block_count or set(types) - {"hybrid", "hybrid_sliding"}:
            raise ValueError("Inkling2: invalid layer_types")
        self.is_swa = [t == "hybrid_sliding" for t in types]

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        tensors = super().index_tensors(remote_hf_model_id)
        if not self.n_draft or self.no_mtp:
            return tensors

        type(self)._indexing_draft = True
        try:
            draft = self.index_local_tensors("mtp")
        finally:
            type(self)._indexing_draft = False

        for name, gen in draft.items():
            if name in tensors:
                raise ValueError(f"Inkling2: drafter tensor collides with a trunk tensor: {name}")
            tensors[name] = gen

        logger.info(f"Inkling2: embedding {len(draft)} drafter tensors as nextn layers, "
                    f"blk.{self._draft_beg}..{self._draft_beg + self.n_draft - 1}")
        return tensors

    @classmethod
    def _draft_name(cls, name: str) -> str:
        # embedded nextn layers use absolute block ids, as llama.cpp addresses them; globals get the
        # nextn namespace. Names are otherwise the checkpoint's own, so the mapping is auditable.
        while name.startswith("mtp."):
            name = name[len("mtp."):]
        if name.startswith("layers."):
            _, stage, rest = name.split(".", 2)
            bid = int(stage)
            if not 0 <= bid < cls._draft_count:
                raise ValueError(f"Inkling2: unexpected drafter stage in {name}")
            return f"blk.{cls._draft_beg + bid}.nextn.{rest}"
        if name == "fuse_weight":
            return "nextn.fuse.weight"         # DFlare: per-stage weights over the target taps
        if name == "mask_embed":
            return "nextn.mask_embed.weight"   # block-diffusion noise slot embedding
        if name == "norm.weight":
            return "nextn.output_norm.weight"
        if name.startswith("markov_head."):
            return f"nextn.{name}"
        raise ValueError(f"Inkling2: unmapped drafter tensor {name}")

    def dequant_model(self):
        for name in list(self.model_tensors):
            if not name.endswith((".w13_weight", ".w2_weight")):
                continue
            weight = self.model_tensors[name]
            tensor = weight()
            companions = [name + suffix for suffix in ("_scale", "_amax", "_outer_scale_denominator")]
            if tensor.dtype.is_floating_point and tensor.element_size() > 1:
                if any(n in self.model_tensors for n in companions):
                    raise ValueError(f"Inkling2: floating-point {name} has packed quantization companions")
                continue
            config = self._expert_quant.get(name, {})
            scheme = config.get("weight_transform")
            if scheme not in ("nvfp4_nearest", "mxfp8_nearest"):
                raise ValueError(f"Inkling2: {name}: missing or unsupported hf_quant_config.json scheme {scheme!r}")
            nvfp4 = scheme == "nvfp4_nearest"
            if nvfp4 and config.get("weight_scale_format") != "amax_denominator":
                raise ValueError(f"Inkling2: {name}: expected NVFP4 amax_denominator scale format")
            required = companions if nvfp4 else companions[:1]
            for companion in required:
                if companion not in self.model_tensors:
                    raise ValueError(f"Inkling2: {name}: missing companion {companion}")
            if not nvfp4 and any(n in self.model_tensors for n in companions[1:]):
                raise ValueError(f"Inkling2: {name}: MXFP8 must not have amax or outer_scale_denominator")
            scale = self.model_tensors[companions[0]]
            scale_tensor = scale()
            weight_dtype, scale_dtype = (torch.uint8, torch.float8_e4m3fn) if nvfp4 else (torch.float8_e4m3fn, torch.uint8)
            if tensor.dtype != weight_dtype or scale_tensor.dtype != scale_dtype:
                raise ValueError(f"Inkling2: {name}: expected {weight_dtype} weights and {scale_dtype} scales, got {tensor.dtype} and {scale_tensor.dtype}")
            experts = self.hparams["n_routed_experts"]
            hidden = self.hparams["hidden_size"]
            inter = self.find_hparam(["moe_intermediate_size", "intermediate_size"])
            w13 = name.endswith(".w13_weight")
            rows, cols = (2 * inter, hidden) if w13 else (hidden, inter)
            shape = (experts, rows, cols)
            if min(shape) <= 0 or cols % (16 if nvfp4 else 32) or tuple(tensor.shape) != (experts, rows, cols // (2 if nvfp4 else 1)):
                raise ValueError(f"Inkling2: {name}: invalid packed shape {tuple(tensor.shape)} for logical shape {shape}")
            ss = tuple(scale_tensor.shape)
            if len(ss) != 3 or ss[0] != experts or min(ss) <= 0 or rows % ss[1] or cols % ss[2]:
                raise ValueError(f"Inkling2: {name}_scale: shape {ss} does not divide logical shape {shape}")
            block = (rows // ss[1], cols // ss[2])
            if block != (1, 16 if nvfp4 else 32):
                raise ValueError(f"Inkling2: {name}_scale: unsupported stored block shape {block}")
            if list(block) != config.get("weight_block_shape"):
                logger.warning("Inkling2: %s: hf_quant_config.json weight_block_shape %s conflicts with stored weight/scale shapes %s/%s; using %s", name, config.get("weight_block_shape"), tuple(tensor.shape), ss, list(block))
            amax = denominator = None
            if nvfp4:
                amax, denominator = (self.model_tensors[n] for n in companions[1:])
                for companion, gen, expected in ((companions[1], amax, (experts,)), (companions[2], denominator, (1,))):
                    t = gen()
                    if t.dtype != torch.float32 or tuple(t.shape) != expected:
                        raise ValueError(f"Inkling2: {companion}: expected F32 {expected}, got {t.dtype} {tuple(t.shape)}")
            native = self.native_experts and (not nvfp4 or cols % _QK_NVFP4 == 0)
            if self.native_experts and not native:
                logger.warning("Inkling2: %s: %d columns is not a multiple of %d, dequantizing instead of keeping NVFP4", name, cols, _QK_NVFP4)
            if native:
                if nvfp4:
                    self.add_native_nvfp4_experts(name, weight, scale, amax, denominator, shape, w13)
                else:
                    self.add_native_mxfp8_experts(name, weight, scale, shape, w13)
                del self.model_tensors[name]
                for companion in required:
                    del self.model_tensors[companion]
                continue
            chunks: list[Callable[[], np.ndarray]] = [
                partial(_dequant_expert_chunk, weight, scale, amax, denominator, name, expert, row_slice)
                for expert, row_slice in _expert_row_slices(experts, rows, w13)
            ]
            self._packed_experts[name] = gguf.LazyChunkedTensor(chunks, shape, np.float32)
            self.model_tensors[name] = lambda shape=shape: torch.empty(shape, dtype=torch.float32, device="meta")
            for companion in required:
                del self.model_tensors[companion]
        for name, gen in self.model_tensors.items():
            t = gen()
            if not t.dtype.is_floating_point or t.element_size() == 1 or name.endswith(("_weight_scale", "_weight_amax", "_weight_outer_scale_denominator")):
                raise ValueError(f"Inkling2: unexpected packed tensor or orphan companion {name}")

    def add_native_mxfp8_experts(self, name: str, weight: Callable[[], Tensor], scale: Callable[[], Tensor], shape: tuple[int, int, int], w13: bool) -> None:
        # Writes one stacked [n_expert, rows, cols] MXFP8 tensor straight to the gguf writer,
        # chunked so only one expert row block is materialized while it is streamed.
        experts, rows, cols = shape
        byte_shape = (experts, rows, cols // _QK_MXFP8 * (_QK_MXFP8 + 1))
        chunks: list[Callable[[], np.ndarray]] = [
            partial(_pack_mxfp8_chunk, weight, scale, name, expert, row_slice)
            for expert, row_slice in _expert_row_slices(experts, rows, w13)
        ]
        target = name.replace(".w13_weight", ".gate_up_proj").replace(".w2_weight", ".down_proj")
        new_name = self.map_tensor_name(target, try_suffixes=()) + ".weight"
        logger.info("Inkling2: %s: %s, kept as MXFP8, shape = %s", new_name, name, tuple(reversed(shape)))
        self.gguf_writer.add_tensor(new_name, gguf.LazyChunkedTensor(chunks, byte_shape, np.uint8), raw_dtype=gguf.GGMLQuantizationType.MXFP8)

    def add_native_nvfp4_experts(self, name: str, weight: Callable[[], Tensor], scale: Callable[[], Tensor], amax: Callable[[], Tensor], denominator: Callable[[], Tensor], shape: tuple[int, int, int], w13: bool) -> None:
        # Same streaming as the MXFP8 export, plus the per-expert global factor that ggml NVFP4 has no
        # room for: it goes to the .scale sidecar the runtime multiplies the expert output by.
        experts, rows, cols = shape
        block = _QK_NVFP4 // 2 + _QK_NVFP4 // _QK_NVFP4_SUB
        byte_shape = (experts, rows, cols // _QK_NVFP4 * block)
        chunks: list[Callable[[], np.ndarray]] = [
            partial(_pack_nvfp4_chunk, weight, scale, name, expert, row_slice)
            for expert, row_slice in _expert_row_slices(experts, rows, w13)
        ]
        target = name.replace(".w13_weight", ".gate_up_proj").replace(".w2_weight", ".down_proj")
        new_name = self.map_tensor_name(target, try_suffixes=()) + ".weight"
        logger.info("Inkling2: %s: %s, kept as NVFP4, shape = %s", new_name, name, tuple(reversed(shape)))
        self.gguf_writer.add_tensor(new_name, gguf.LazyChunkedTensor(chunks, byte_shape, np.uint8), raw_dtype=gguf.GGMLQuantizationType.NVFP4)
        self.add_nvfp4_expert_scale(new_name, amax, denominator, experts, name)

    def add_nvfp4_expert_scale(self, new_name: str, amax: Callable[[], Tensor], denominator: Callable[[], Tensor], experts: int, name: str) -> None:
        # amax / outer_scale_denominator is not a power of two in this checkpoint, so it cannot be folded
        # into the ue4m3 block scales; divide in float32 exactly like the dequant path does per expert.
        a = LazyTorchTensor.to_eager(amax()).numpy()
        d = LazyTorchTensor.to_eager(denominator()).numpy()
        if a.size != experts or d.size != 1:
            raise ValueError(f"Inkling2: {name}: expected amax ({experts},) and outer_scale_denominator (1,), got {a.size} and {d.size}")
        if not np.isfinite(a).all() or (a < 0).any():
            raise ValueError(f"Inkling2: {name}_amax: expected finite nonnegative values")
        if not np.isfinite(d).all() or (d <= 0).any():
            raise ValueError(f"Inkling2: {name}_outer_scale_denominator: expected finite positive value")
        factor = a / d.reshape(())
        logger.info("Inkling2: %s: per-expert scale, shape = [%d]", new_name.replace(".weight", ".scale"), experts)
        self.gguf_writer.add_tensor(new_name.replace(".weight", ".scale"), factor)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if cls._indexing_draft:
            return cls._draft_name(name), gen
        if any(p in name for p in ("visual.", "vision_tower.", "mtp.", "markov_head", "nextn_predict")):
            return None
        name = name.replace("model.language_model.", "model.").replace("model.llm.", "model.")
        # the mapping table cannot shift a block index: attn_queries.{i} feeds block i + 1
        name = re.sub(r"attn_res\.attn_queries\.(\d+)", lambda m: f"layers.{int(m[1]) + 1}.attention_input.query", name)
        # the table maps a module, not its bias: gate.weight is the router, gate.bias the score correction
        name = name.replace("mlp.gate.bias", "mlp.gate.e_score_correction_bias")
        return name, gen

    def get_vocab_base(self):
        tokens, types, pre = super().get_vocab_base()
        n = self.hparams["unpadded_vocab_size"]
        return tokens[:n], types[:n], pre

    def set_vocab(self):
        tokens, types, pre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(pre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(types)
        special = gguf.SpecialVocab(self.dir_model, load_merges=True)
        eos = self.hparams["eos_token_id"]
        special.special_token_ids["eos"] = eos
        special.special_token_ids["eot"] = eos
        special.add_special_token["bos"] = False
        special.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        hp = self.hparams
        # the drafter widens the per-layer ffn; the expert size stays the trunk's scalar
        n_ff_expert = hp["intermediate_size"]
        if self.n_draft:
            md = self.mtp
            swa = (hp.get("swa_num_attention_heads", hp["num_attention_heads"]),
                   hp.get("swa_num_key_value_heads", hp["num_key_value_heads"]),
                   hp.get("swa_head_dim", hp["head_dim"]))
            draft = (md["num_attention_heads"], md["num_key_value_heads"], md["head_dim"])
            if draft != swa:
                raise ValueError(f"Inkling2: drafter heads {draft} != trunk swa heads {swa}, "
                                 f"the per-layer arrays below assume they agree")
            if md.get("hidden_size") != hp["hidden_size"]:
                raise ValueError("Inkling2: the drafter must share the trunk hidden size")
            if md.get("rms_norm_eps") != hp.get("rms_norm_eps"):
                raise ValueError("Inkling2: drafter rms_norm_eps must match the trunk, the loader has one array")
            if md.get("sliding_window_size") != hp.get("sliding_window_size"):
                raise ValueError("Inkling2: drafter sliding_window_size must match the trunk")
            trunk_theta = hp.get("rope_theta")
            if trunk_theta is not None and md.get("rope_theta") not in (None, trunk_theta):
                raise ValueError(f"Inkling2: drafter rope_theta {md['rope_theta']} != trunk {trunk_theta}")
            # local_layer_ids is authoritative: draft layer 0 and 2 window, layer 1 attends fully.
            # Kept out of the trunk's pattern array: inkling2 requires it to be exactly n_layer() long
            local = md.get("local_layer_ids", [])
            self.draft_swa = [i in local for i in range(self.n_draft)]
            hp["intermediate_size"] = [n_ff_expert] * self.block_count + [md["intermediate_size"]] * self.n_draft
            self.block_count += self.n_draft
        hp["num_attention_heads"] = [hp.get("swa_num_attention_heads", hp["num_attention_heads"]) if s else hp["num_attention_heads"] for s in self.is_swa]
        hp["num_key_value_heads"] = [hp.get("swa_num_key_value_heads", hp["num_key_value_heads"]) if s else hp["num_key_value_heads"] for s in self.is_swa]
        if self.n_draft:
            # these arrays are read with n_layer_all (src/llama-model.cpp:1608-1617), so the draft layers
            # need entries of their own; sliding_window_pattern is the exception - it is sized n_layer()
            hp["num_attention_heads"] = hp["num_attention_heads"] + [self.mtp["num_attention_heads"]] * self.n_draft
            hp["num_key_value_heads"] = hp["num_key_value_heads"] + [self.mtp["num_key_value_heads"]] * self.n_draft
        for name in ("n_routed_experts", "num_experts_per_tok"):
            if name not in hp:
                raise KeyError(f"Inkling2: missing {name}")
        super().set_gguf_parameters()
        w = self.gguf_writer
        w.add_vocab_size(hp["vocab_size"])
        w.add_vocab_size_unpadded(hp["unpadded_vocab_size"])
        w.add_expert_feed_forward_length(n_ff_expert)
        w.add_key_length_swa(hp.get("swa_head_dim", hp["head_dim"]))
        w.add_value_length_swa(hp.get("swa_head_dim", hp["head_dim"]))
        w.add_rope_dimension_count(0)
        w.add_sliding_window(hp["sliding_window_size"])
        w.add_sliding_window_pattern(self.is_swa)
        w.add_shortconv_l_cache(self.find_hparam(["conv_kernel_size", "sconv_kernel_size"]))
        w.add_expert_weights_scale(hp["route_scale"])
        if self.n_draft:
            md = self.mtp
            w.add_nextn_predict_layers(self.n_draft)
            p = f"{gguf.MODEL_ARCH_NAMES[self.model_arch]}.mtp"  # self.model_name is unset for this converter
            w.add_uint32(f"{p}.block_size", md["block_size"])
            w.add_uint32(f"{p}.markov_rank", md["markov_rank"])
            w.add_string(f"{p}.markov_head_type", md["markov_head_type"])
            w.add_array(f"{p}.target_layer_ids", md["target_layer_ids"])
            # uint32, not bool: llama_model_loader::get_arr<vector<uint32_t>> rejects a BOOL array
            w.add_array(f"{p}.attention.sliding_window_pattern", [int(x) for x in self.draft_swa])
            # the drafter's MLP is dense and it has no experts: the per-layer expert arrays are read with
            # n_layer_all, so the draft entries must be present and zero or generic sizing paths treat
            # the stages as expert layers
            n_trunk = self._draft_beg
            w.add_expert_used_count([hp["num_experts_per_tok"]] * n_trunk + [0] * self.n_draft)
            w.add_expert_feed_forward_length([n_ff_expert] * n_trunk + [0] * self.n_draft)
            if md.get("rope_theta") is not None:
                # the trunk carries no rope at all (rope_dimension_count 0, absolute positions come from
                # the relative bias table) while the drafter was trained with theta 1e6. The trunk's
                # scalar is absent, so the draft graph must not silently inherit the loader default.
                w.add_float32(f"{p}.rope_frequency_base", md["rope_theta"])
        w.add_activation_situ_beta(hp["mlp_gate_softcap"])
        w.add_activation_situ_linear_beta(hp["mlp_up_softcap"])
        w.add_attn_res_block_size(hp["n_layers_per_attn_res_block"])
        w.add_final_logit_softcapping(hp["final_logit_softcapping"])
        w.add_relative_length(hp["d_rel"])
        w.add_relative_extent(hp["rel_extent"])
        w.add_qk_clip(hp["qk_l2_clip_threshold"])
        w.add_relative_clip(hp["rel_l2_clip_threshold"])
        w.add_log_scale_floor(hp["log_scaling_n_floor"])
        w.add_log_scale_alpha(hp["log_scaling_alpha"])

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        if new_name.endswith(("_conv.weight", "_clip.weight", "_res_query.weight", "attn_rel_bias.weight")):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("nextn.") or ".nextn." in name:
            # the drafter keeps the name the checkpoint gave it and its shipped dtype (bf16 per
            # mtp_config.torch_dtype); the trunk tensor map has no entries for these and must not guess
            yield name, data_torch
            return
        if name in self._packed_experts:
            target = name.replace(".w13_weight", ".gate_up_proj").replace(".w2_weight", ".down_proj")
            yield self.map_tensor_name(target, try_suffixes=()) + ".weight", cast("Tensor", self._packed_experts[name])
            return
        if name.endswith(".w13_weight"):
            shape = data_torch.shape
            data_torch = data_torch.reshape(shape[0], shape[1] // 2, 2, shape[2]).transpose(1, 2).reshape(shape)
            name = name.replace(".w13_weight", ".gate_up_proj")
        name = name.replace(".w2_weight", ".down_proj")
        for static, conv in ((".static_weight", ".conv1d.weight"), (".static_w", ".weight")):
            if name.endswith(static):
                data_torch = data_torch.flip(0).T.contiguous()
                name = name.removesuffix(static) + conv
        if name.endswith((".conv1d.weight", "sconv.weight")):
            data_torch = data_torch.squeeze(1) if data_torch.ndim == 3 else data_torch
        elif name.endswith(".rel_logits_proj.proj"):
            data_torch = data_torch.T.contiguous()
        # a name the table maps without a suffix is a bare parameter
        new_name = self.tensor_map.get_name(name)
        if new_name is not None:
            suffix = ".bias" if name.endswith(".e_score_correction_bias") else ".weight"
            yield new_name + suffix, data_torch
        else:
            yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Inkling2LoopForConditionalGeneration", "Inkling2LoopForCausalLM")
class Inkling2LoopModel(Inkling2Model):
    model_arch = gguf.MODEL_ARCH.INKLING2_LOOP

    # raw checkpoint and transformers names of the cross-pass message queries
    _query_kinds = {
        "model.middle_cycle_cross_pass_queries": "q",
        "model.middle_cycle_cross_pass_kv_queries": "kv",
        "model.cross_pass_messages.queries": "q",
        "model.cross_pass_messages.kv_queries": "kv",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._queries: dict[str, Tensor] = {}
        self.loop = self.hparams["loop_config"]
        if set(self.loop) != {"core_start", "core_end", "num_passes", "gate_rank", "is_varloop"}:
            raise ValueError(f"Inkling2Loop: unsupported loop_config keys {sorted(self.loop)}")
        start, end, block = self.loop["core_start"], self.loop["core_end"], self.hparams["n_layers_per_attn_res_block"]
        # a fixed loop needs at least 2 passes, a variable loop at least 1
        min_passes = 1 if self.loop["is_varloop"] else 2
        if not 0 < start < end <= self.block_count or start % block or end % block or self.loop["num_passes"] < min_passes or self.loop["gate_rank"] < 1:
            raise ValueError(f"Inkling2Loop: invalid loop_config {self.loop}")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_loop_start(self.loop["core_start"])
        self.gguf_writer.add_loop_end(self.loop["core_end"])
        self.gguf_writer.add_loop_count(self.loop["num_passes"])
        self.gguf_writer.add_loop_gate_rank(self.loop["gate_rank"])
        self.gguf_writer.add_loop_varying(self.loop["is_varloop"])

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        # the reference does not quantize the loop tensors
        if new_name.endswith(("loop_msg_query.weight", "_ctrl_down.weight", "_ctrl_gain.weight", "_ctrl_forget.weight")):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name not in self._query_kinds:
            yield from super().modify_tensors(data_torch, name, bid)
            return
        self._queries[self._query_kinds[name]] = data_torch
        if len(self._queries) < 2:
            return
        q, kv = self._queries["q"], self._queries["kv"]
        start = self.loop["core_start"]
        if q.shape[0] != 2 * (self.loop["core_end"] - start) or kv.shape != q.shape:
            raise ValueError(f"Inkling2Loop: cross-pass queries {tuple(q.shape)} and {tuple(kv.shape)} do not match the loop core")
        # core layer start + i mixes its q, k, v and mlp messages with these four queries
        for i in range(q.shape[0] // 2):
            query = torch.stack([q[2 * i], kv[2 * i], kv[2 * i + 1], q[2 * i + 1]])
            yield self.format_tensor_name(gguf.MODEL_TENSOR.LOOP_MSG_QUERY, start + i, ".weight"), query


@ModelBase.register("Inkling2ForConditionalGeneration", "Inkling2LoopForConditionalGeneration")
class Inkling2VisionModel(MmprojModel):
    def get_vision_config(self):
        config = self.global_config.get("vision_config")
        if not config:
            return None
        config = dict(config)
        config["patch_size"] = config.get("native_patch_size", config.get("patch_size", 20))
        config["spatial_merge_size"] = config.get("adapter_merge_size", config.get("spatial_merge_size", 2))
        config["intermediate_size"] = config.get("intermediate_size") or int(config["hidden_size"] * config.get("mlp_ratio", 4.0))
        config["image_size"] = config["posemb_grid_size"] * config["patch_size"]
        if config.get("num_channels", 3) != 3 or config.get("hidden_act", "gelu") != "gelu":
            raise ValueError("Inkling2 vision requires RGB patches and exact GELU")
        return config

    def set_gguf_parameters(self):
        self.preprocessor_config.setdefault("image_mean", [0.5, 0.5, 0.5])
        self.preprocessor_config.setdefault("image_std", [0.5, 0.5, 0.5])
        super().set_gguf_parameters()
        config = self.hparams
        rope = config.get("rope_parameters") or {"rope_type": "axial", "rope_theta": 10000.0}
        if rope["rope_type"] != "axial" or rope["rope_theta"] != 10000.0:
            raise ValueError("Inkling2 vision requires axial RoPE with theta 10000")
        w = self.gguf_writer
        w.add_clip_vision_projector_type(gguf.VisionProjectorType.INKLING2V)
        w.add_vision_attention_layernorm_eps(config["layer_norm_eps"])
        w.add_vision_spatial_merge_size(config["spatial_merge_size"])

    def dequant_model(self):
        # vision tensors are BF16, only the text experts are quantized
        return

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        if not name.startswith(("model.visual.", "model.vision_tower.")):
            return None
        return name, gen

    # mm.{bid} takes the projector layer index, the mapping table cannot derive it from these names
    _projector_layers = {"model.visual.proj_in": 1, "model.visual.proj_out": 2, "model.vision_tower.merger.linear_1": 1, "model.vision_tower.merger.linear_2": 2}

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        module, _, suffix = name.rpartition(".")
        if module in self._projector_layers:
            yield self.format_tensor_name(gguf.MODEL_TENSOR.V_MMPROJ, self._projector_layers[module], "." + suffix), data_torch
            return
        # lay out the linear patch embedding (HWC patches) as a conv2d for build_inp()
        if name.endswith(".patch_embedding.weight"):
            ps = self.hparams["patch_size"]
            data_torch = data_torch.reshape(data_torch.shape[0], ps, ps, 3).permute(0, 3, 1, 2)
        # these are bare parameters, they have no .weight
        if name.endswith(("position_embedding", ".lambda_1", ".lambda_2", ".gamma")):
            name = name.removesuffix(".gamma") + ".weight"
        yield from super().modify_tensors(data_torch, name, bid)
