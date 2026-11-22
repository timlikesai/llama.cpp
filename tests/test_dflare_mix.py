"""Numeric oracle for the DFlare tap mixing of Inkling DSpark drafts.

The reference is the vLLM implementation that docs/mxfp8/33-drafter-reference-semantics.md transcribes:

    fuse_alpha = softmax(fuse_weight.float(), dim=-1)        # one row of tap weights per mix
    mixed      = sum_t fuse_alpha[row][t] * features[t]      # features: (rows, taps, hidden)
    out        = mixed * rsqrt(mean(mixed^2) + eps) * norm.weight

tests/test-dflare-mix.cpp runs the ggml graph (llama_model_dflash::build_dflare_alpha, build_dflare_taps
and build_dflare_mix) on the same inputs and compares against the constants embedded there, so the two must stay in sync. Both
sides draw the inputs from the same LCG in the same order, which the input_sum constant pins.

The reference accumulates in float64 while the graph is fp32, so the C++ check uses a tolerance instead of
bit equality. vLLM stores fuse_alpha in the dtype of fuse_weight (bf16); that rounding is not reproduced by
the draft either, so the oracle keeps the weights in the higher precision.

Run from the repository root:
    python3 tests/test_dflare_mix.py                                          # print the C++ constants
    python3 -m unittest discover -s tests -p 'test_dflare_mix.py' -v          # check what is pasted there
"""

import re
import unittest

import numpy as np

MASK64 = 0xFFFFFFFFFFFFFFFF

# the same constants as frand() in tests/test-dflare-mix.cpp
LCG_MULT = 6364136223846793005
LCG_ADD = 1442695040888963407
LCG_SEED = 0x12345678

# name, n_taps, n_embd, n_tokens, n_mix, eps, element-wise reference
# 'full' cases compare every output, the others compare mean, meansq, max and four spot values
CASES = (
    ("tiny",    4,    6, 2, 4, np.float32(1e-6), True),
    ("aligned", 4,   64, 3, 4, np.float32(1e-6), False),
    ("inkling", 4, 2304, 5, 4, np.float32(1e-6), False),
)

SPOT_FRACTIONS = (0.0, 1 / 3, 2 / 3, 1.0)

DECL_RE = re.compile(
    r"static const (?:float|double) k_ref_(?P<name>\w+?)_(?P<kind>alpha|mix\d+|stat)"
    r"(?:\s*\[[^\]]*\])*\s*=\s*\{")

SCALAR_RE = re.compile(r"static const double k_ref_(?P<name>\w+?)_in = (?P<value>[-0-9.eE+]+);")


class Rand:
    """frand() of the C++ test: a 64 bit LCG, the top 24 bits mapped to [-1, 1) in fp32."""

    def __init__(self):
        self.state = LCG_SEED

    def draw(self, n):
        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            self.state = (self.state * LCG_MULT + LCG_ADD) & MASK64
            v = np.float32((self.state >> 33) & 0xFFFFFF) / np.float32(0x800000)
            out[i] = v - np.float32(1.0)
        return out


def reference(rng, n_taps, n_embd, n_tokens, n_mix, eps):
    fuse = rng.draw(n_taps * n_mix).astype(np.float64).reshape(n_mix, n_taps)
    norm = rng.draw(n_embd * n_mix).astype(np.float64).reshape(n_mix, n_embd)
    feats = rng.draw(n_taps * n_embd * n_tokens).astype(np.float64).reshape(n_tokens, n_taps, n_embd)

    input_sum = float(fuse.sum() + norm.sum() + feats.sum())

    a = np.exp(fuse - fuse.max(axis=-1, keepdims=True))
    alpha = a / a.sum(axis=-1, keepdims=True)

    # ggml stores the mix weights as {n_taps, n_mix}, which is the torch {n_mix, n_taps} row-major buffer
    mixed = np.einsum('mt,itd->mid', alpha, feats)
    scale = 1.0 / np.sqrt(np.mean(mixed * mixed, axis=-1, keepdims=True) + eps)
    out = (mixed * scale) * norm[:, None, :]

    # a ggml {n_embd, n_tokens} tensor flattens as i * n_embd + e, the C order of out[m]
    return input_sum, alpha.reshape(-1), out.reshape(n_mix, n_embd * n_tokens)


def stats(row):
    n = len(row)
    return [row.mean(), np.mean(row * row), row.max()] + [row[int(f * (n - 1))] for f in SPOT_FRACTIONS]


def expected():
    rng = Rand()
    cases = []
    for name, n_taps, n_embd, n_tokens, n_mix, eps, full in CASES:
        input_sum, alpha, out = reference(rng, n_taps, n_embd, n_tokens, n_mix, eps)
        cases.append({
            "name":      name,
            "n_taps":    n_taps,
            "n_mix":     n_mix,
            "input_sum": input_sum,
            "alpha":     alpha,
            "full":      full,
            "out":       out if full else np.array([stats(row) for row in out]),
        })
    return cases


def fmt(values):
    return ", ".join("%.9gf" % float(v) for v in values)


def emit():
    lines = []
    for c in expected():
        name = c["name"]
        lines.append("static const double k_ref_%s_in = %.9g;" % (name, c["input_sum"]))
        lines.append("static const float k_ref_%s_alpha[] = { %s };" % (name, fmt(c["alpha"])))
        if c["full"]:
            for m, row in enumerate(c["out"]):
                lines.append("static const float k_ref_%s_mix%d[] = { %s };" % (name, m, fmt(row)))
        else:
            lines.append("static const float k_ref_%s_stat[][7] = {" % name)
            for row in c["out"]:
                lines.append("    { %s }," % fmt(row))
            lines.append("};")
        lines.append("")
    return "\n".join(lines)


def pasted(source):
    blocks = {}
    for match in SCALAR_RE.finditer(source):
        blocks[(match.group("name"), "in")] = [[float(match.group("value"))]]

    for match in DECL_RE.finditer(source):
        depth = 1
        i = match.end()
        while depth > 1 or source[i] != "}":
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
            i += 1
        end = i
        body = source[match.end():end].replace("{", " ").replace("}", " ").replace(",", " ")
        values = [float(v.rstrip("f")) for v in body.split()]
        blocks.setdefault((match.group("name"), match.group("kind")), []).append(values)
    return blocks


class TestDflareOracle(unittest.TestCase):
    """Recompute the oracle and check that the constants in the C++ test still match it."""

    def test_pasted_constants(self):
        with open("tests/test-dflare-mix.cpp", "r", encoding="utf-8") as f:
            source = f.read()

        blocks = pasted(source)
        self.assertTrue(blocks, "no oracle constants found in tests/test-dflare-mix.cpp")

        for case in expected():
            name = case["name"]
            self.assertIn((name, "in"), blocks, "missing k_ref_%s_in" % name)
            self.assertAlmostEqual(blocks[(name, "in")][0][0], case["input_sum"], delta=1e-6)
            alpha = np.array(blocks[(name, "alpha")][0], dtype=np.float64)
            np.testing.assert_allclose(alpha, case["alpha"], rtol=0, atol=1e-7)

            if case["full"]:
                got = np.array([blocks[(name, "mix%d" % m)][0] for m in range(case["n_mix"])],
                               dtype=np.float64)
            else:
                got = np.array(blocks[(name, "stat")][0], dtype=np.float64).reshape(case["n_mix"], 7)

            self.assertEqual(got.shape, case["out"].shape)
            np.testing.assert_allclose(got, case["out"], rtol=0, atol=1e-7)


if __name__ == "__main__":
    print(emit())
