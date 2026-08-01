#!/usr/bin/env python3
from pathlib import Path
import re

path = Path("ggml/src/ggml-cuda/tiered.cu")
text = path.read_text()
pattern = re.compile(
    r"    std::vector<size_t> chunks;\n"
    r"    chunks\.reserve\(host_ids\.size\(\)\);\n"
    r"    const size_t granularity = state->sparse_granularity;\n\n"
    r"    for \(const int32_t expert : host_ids\) \{.*?"
    r"    chunks\.erase\(std::unique\(chunks\.begin\(\), chunks\.end\(\)\), chunks\.end\(\)\);\n",
    re.DOTALL,
)
replacement = """    for (const int32_t expert : host_ids) {
        if (expert < 0 || expert >= tensor->ne[2]) {
            throw std::runtime_error(\"MUL_MAT_ID produced an out-of-range expert id\");
        }
    }

    // Correctness fallback for pre-Ampere/Turing-class GPUs: map the complete
    // backing tensor for the duration of this MUL_MAT_ID node. The MMVQ kernels
    // may issue vectorized reads outside the exact selected-expert byte interval,
    // while still remaining inside the tensor allocation. Mapping the whole
    // tensor prevents those reads from touching an unmapped VMM page.
    std::vector<size_t> chunks;
    const size_t granularity = state->sparse_granularity;
    chunks.reserve(state->sparse_size / granularity);
    for (size_t offset = 0; offset < state->sparse_size; offset += granularity) {
        chunks.push_back(offset);
    }
"""
new_text, count = pattern.subn(replacement, text, count=1)
if count != 1:
    raise SystemExit("tiered.cu did not match the expected selective-mapping block")
path.write_text(new_text)
print(f"patched {path}")
