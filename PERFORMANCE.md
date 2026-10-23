# Recorded native run and attention change

The uploaded Kaggle notebook provides these measurements:

| Item | Recorded value |
|---|---:|
| Hardware | Tesla T4 ×2 |
| Runtime | Python 3.13.15, PyTorch 2.11.0+cu128 |
| Precision | FP16 transformer (6.76 GiB resident) |
| Prompt tokens | 1,324 |
| Requested music-token maximum | 9,000 (360 seconds) |
| Static AR capacity | 10,324 |
| AR prefill | 1.494 seconds |
| First token | 1.689 seconds |
| AR outputs, including end token | 5,977 |
| Music-content tokens | 5,976 (239.04 seconds represented) |
| AR time | 715.418 seconds (11m55s) |
| AR throughput | 8.355 outputs/second |
| Acoustic synthesis | Approximately 277 seconds, 16 Euler steps |
| VAE decode | Approximately 7.5 seconds, 12 tiles |

The session duration also includes setup, downloads, UI and other overhead. The
trace ends after decoding and does not contain a saved audio file header or the
UI's final export response. Token duration is not a claim that this attachment
contains recoverable audio. There is no recorded capture-failure message; the
old `execution=cuda_graph` label alone cannot rule out fallback because that
label was faulty. Revision 4 separately corrected execution/progress reporting.

## Backend restriction verified from source

The model has 16 query heads and 8 KV heads. The native graph and acoustic
attention call SDPA with `enable_gqa=True`. PyTorch's documentation limits CUDA
GQA to FlashAttention/math, and its 2.11 CUDA source restricts FlashAttention
and cuDNN attention to SM80 or newer. T4 is SM75. The native GQA call therefore
cannot use those fused alternatives on this hardware; memory-efficient SDPA
requires equal head counts. This is source evidence, not a GPU profiler trace
or proof of the complete difference from an earlier reported fast song.

- [PyTorch SDPA documentation](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [PyTorch 2.11 backend checks](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp)
- [Pinned native package/model](https://huggingface.co/m-a-p/YuE2-3B/tree/8f24312f187763b9854adeec87e437e2b03bbff1)

## Revision 5 implementation

`native_yue2/attention.py` represents KV heads as additional batches and expands
within-head groups using views. Equal query/KV head counts let the efficient
backend be considered without `enable_gqa`. Masks preserve their logical values;
row-stride padding is outside the logical key length. The result is restored to
the original head layout. For the default single guidance branch, tests verify
that K/V views reuse existing storage rather than repeat the cached data.

Each tensor shape/stride combination is checked for CUDA eligibility. The first
actual efficient-kernel dispatch is checked before caching the decision. An
unsupported/refused shape logs its reason and retains native SDPA; OOM is raised.
Confirmed selections and native fallbacks are reported. Only the native modules'
functional namespaces are wrapped; global `torch.nn.functional` is unchanged.

Eight attention tests passed on CPU, including seven numerical shape/mask
subcases and emulated dispatch-control tests. Twelve native/UI/export/recovery
checks also passed. No actual T4 acceleration, graph compatibility, memory peak,
long-song acoustic quality or completion time has been validated for this change.
The earlier three-minute/under-two-minute claim lacks a comparable saved run log.
No repeat Kaggle diagnostic generation was performed or requested for these checks.
