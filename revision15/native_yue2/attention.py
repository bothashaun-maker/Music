"""Equivalent grouped attention that can use PyTorch's efficient kernel on T4.

CUDA GQA via enable_gqa=True supports FlashAttention/math, excluding the
memory-efficient backend. T4 cannot use FlashAttention. Grouping KV heads into
batch views exposes ordinary equal-head SDPA to the efficient backend instead.
Eligibility and the first actual dispatch are checked; unsupported cases retain
native GQA. No model weights, token budgets or solver settings are changed.
"""
import torch
import torch.nn.functional as functional

DECISIONS = {}
STATUS = {'efficient_grouped_shapes': 0, 'native_fallback_shapes': 0}


def grouped_views(query, key, value, attn_mask=None):
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError('Expected [batch, heads, tokens, channels]')
    batch, heads, length, width = query.shape
    kv_heads, source = key.shape[1:3]
    if (heads % kv_heads or key.shape[0] != batch or value.shape[:3] != key.shape[:3]
            or key.shape[-1] != width):
        raise ValueError('Invalid grouped-query attention shapes')
    groups = heads // kv_heads
    q = query.reshape(batch * kv_heads, groups, length, width)
    k = key.reshape(batch * kv_heads, 1, source, width).expand(-1, groups, -1, -1)
    v = value.reshape(batch * kv_heads, 1, source, value.shape[-1]).expand(-1, groups, -1, -1)
    bias = None
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            bias = torch.zeros(attn_mask.shape, device=query.device, dtype=query.dtype)
            bias.masked_fill_(~attn_mask, float('-inf'))
        else:
            bias = attn_mask
        bias = torch.broadcast_to(bias, (*bias.shape[:-1], source))
        # CUTLASS attention needs aligned row strides. Keep padding outside the
        # logical key length so the mathematical mask is unchanged.
        padding = (-source) % 8
        if padding:
            bias = functional.pad(bias, (0, padding))[..., :source]
        bias = torch.broadcast_to(bias, (batch, heads, length, source))
        bias = bias.reshape(batch * kv_heads, groups, length, source)
    return q, k, v, bias


def restore_heads(output, query):
    return output.reshape(query.shape[0], query.shape[1], query.shape[2], output.shape[-1])


def sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None, enable_gqa=False):
    kwargs = dict(attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
                  scale=scale, enable_gqa=enable_gqa)
    if not (enable_gqa and query.is_cuda and query.ndim == key.ndim == value.ndim == 4
            and query.dtype == key.dtype == value.dtype == torch.float16 and dropout_p == 0):
        return functional.scaled_dot_product_attention(query, key, value, **kwargs)
    q, k, v, bias = grouped_views(query, key, value, attn_mask)
    signature = (tuple(q.shape), tuple(k.shape), tuple(v.shape), tuple(q.stride()),
                 tuple(k.stride()), tuple(v.stride()),
                 None if bias is None else (tuple(bias.shape), tuple(bias.stride()), bias.dtype),
                 query.device.index, is_causal, scale)
    decision = DECISIONS.get(signature)
    if decision is None:
        parameters = torch.backends.cuda.SDPAParams(q, k, v, bias, dropout_p, is_causal, False)
        decision = torch.backends.cuda.can_use_efficient_attention(parameters)
        if decision:
            try:
                from torch.nn.attention import SDPBackend, sdpa_kernel
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    result = functional.scaled_dot_product_attention(
                        q, k, v, attn_mask=bias, dropout_p=0, is_causal=is_causal, scale=scale,
                        enable_gqa=False)
            except torch.OutOfMemoryError:
                raise
            except RuntimeError as exc:
                print(f'[YuE2 attention] Efficient grouped kernel refused: {exc}; using native SDPA.', flush=True)
                decision = False
            else:
                DECISIONS[signature] = True
                STATUS['efficient_grouped_shapes'] += 1
                print(f'[YuE2 attention] Confirmed efficient grouped kernel: Q={tuple(query.shape)}, K={tuple(key.shape)}', flush=True)
                return restore_heads(result, query)
        DECISIONS[signature] = decision
        if not decision:
            STATUS['native_fallback_shapes'] += 1
            print(f'[YuE2 attention] Efficient grouped kernel unavailable: Q={tuple(query.shape)}, K={tuple(key.shape)}; native SDPA retained.', flush=True)
    if not decision:
        return functional.scaled_dot_product_attention(query, key, value, **kwargs)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        result = functional.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=0, is_causal=is_causal, scale=scale, enable_gqa=False)
    return restore_heads(result, query)


class FunctionalProxy:
    """Override only the native engine modules' F namespaces, never global torch."""
    scaled_dot_product_attention = staticmethod(sdpa)

    def __getattr__(self, name):
        return getattr(functional, name)
