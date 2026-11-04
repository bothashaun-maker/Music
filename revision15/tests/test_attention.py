"""CPU numerical equivalence for the T4 attention layout; no CUDA speed claims."""
import unittest
from unittest.mock import patch
import torch
import torch.nn.functional as F
from native_yue2.attention import grouped_views, restore_heads, sdpa, FunctionalProxy
import native_yue2.attention as attention


class EmulatedCudaTensor(torch.Tensor):
    # Exercise dispatch control with CPU tensors; never claims CUDA execution.
    @property
    def is_cuda(self):
        return True


class AttentionTests(unittest.TestCase):
    def setUp(self):
        attention.DECISIONS.clear()

    def compare(self, batch, heads, kv_heads, length, source, dtype, mask_kind, causal=False, scale=None):
        torch.manual_seed(17)
        # Match the engine's non-contiguous [B,H,T,D] views.
        q = torch.randn(batch, length, heads, 32, dtype=dtype).transpose(1, 2)
        k = torch.randn(batch, source, kv_heads, 32, dtype=dtype).transpose(1, 2)
        v = torch.randn_like(k)
        mask = None
        if mask_kind == 'boolean':
            mask = torch.rand(batch, 1, length, source) > .2
            mask[..., 0] = True
        elif mask_kind == 'additive':
            mask = torch.randn(1, 1, length, source)
            if dtype == torch.float16:
                mask = mask.half()
        elif mask_kind == 'broadcast':
            mask = torch.ones(length, 1, dtype=torch.bool)
        elif mask_kind == 'future':
            mask = torch.arange(source)[None, None, None, :] <= source // 2
        before = None if mask is None else mask.clone()
        expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                 is_causal=causal, scale=scale, enable_gqa=True)
        qq, kk, vv, bias = grouped_views(q, k, v, mask)
        result = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=bias,
                                               is_causal=causal, scale=scale, enable_gqa=False)
        actual = restore_heads(result, q)
        tolerance = .003 if dtype == torch.float16 else 2e-5
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        if mask is not None:
            self.assertTrue(torch.equal(mask, before))

    def test_grouped_equivalence_across_decode_prefill_and_nar_shapes(self):
        cases = [
            (1,16,8,1,10324,torch.float16,'future',False,None),
            (2,16,8,1,33,torch.float16,'future',False,None),
            (1,16,8,256,257,torch.float16,None,False,None),
            (1,8,2,17,17,torch.float32,None,True,None),
            (2,8,2,11,19,torch.float32,'boolean',False,None),
            (1,8,2,9,13,torch.float32,'additive',False,.2),
            (1,8,2,9,13,torch.float32,'broadcast',False,None),
        ]
        for case in cases:
            with self.subTest(case=case):
                self.compare(*case)

    def test_grouped_layout_reuses_kv_storage_for_single_branch(self):
        q = torch.zeros(1, 1, 16, 128).transpose(1, 2)
        k = torch.zeros(1, 10324, 8, 128).transpose(1, 2)
        qq, kk, vv, _ = grouped_views(q, k, k)
        self.assertEqual(kk.untyped_storage().data_ptr(), k.untyped_storage().data_ptr())
        self.assertEqual(vv.untyped_storage().data_ptr(), k.untyped_storage().data_ptr())
        self.assertEqual(kk.stride(1), 0)
        self.assertEqual(qq.shape, (8, 2, 1, 128))

    def test_bias_row_alignment_preserves_logical_shape(self):
        q = torch.randn(1, 8, 3, 32)
        k = torch.randn(1, 2, 11, 32)
        _, _, _, bias = grouped_views(q, k, k, torch.ones(3,11,dtype=torch.bool))
        self.assertEqual(bias.shape, (2,4,3,11))
        self.assertEqual(bias.stride(-2) % 8, 0)

    def test_global_torch_is_not_patched_and_cpu_dispatch_is_unchanged(self):
        original = F.scaled_dot_product_attention
        proxy = FunctionalProxy()
        q = torch.randn(1,8,3,32)
        k = torch.randn(1,2,5,32)
        actual = proxy.scaled_dot_product_attention(q,k,k,enable_gqa=True)
        expected = original(q,k,k,enable_gqa=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertIs(F.scaled_dot_product_attention, original)
        self.assertIs(proxy.linear, F.linear)

    def test_invalid_group_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'Invalid'):
            grouped_views(torch.randn(1,3,2,32),torch.randn(1,2,3,32),torch.randn(1,2,3,32))

    def dispatch_fixture(self):
        q = torch.randn(1,8,1,32,dtype=torch.float16)
        k = torch.randn(1,2,11,32,dtype=torch.float16)
        expected = F.scaled_dot_product_attention(q,k,k,enable_gqa=True)
        return q.as_subclass(EmulatedCudaTensor), k, expected

    def test_eligible_dispatch_is_verified_then_cached(self):
        q,k,expected = self.dispatch_fixture()
        from torch.nn.attention import sdpa_kernel, SDPBackend
        def cpu_math_context(*args, **kwargs):
            return sdpa_kernel(SDPBackend.MATH)
        with patch.object(torch.backends.cuda,'can_use_efficient_attention',return_value=True) as check, \
             patch('torch.nn.attention.sdpa_kernel',side_effect=cpu_math_context):
            first = sdpa(q,k,k,enable_gqa=True)
            second = sdpa(q,k,k,enable_gqa=True)
        self.assertEqual(check.call_count, 1)
        self.assertEqual(list(attention.DECISIONS.values()), [True])
        torch.testing.assert_close(first.as_subclass(torch.Tensor), expected, rtol=.003, atol=.003)
        torch.testing.assert_close(second.as_subclass(torch.Tensor), expected, rtol=.003, atol=.003)

    def test_ineligible_dispatch_retains_native_and_caches_decision(self):
        q,k,expected = self.dispatch_fixture()
        with patch.object(torch.backends.cuda,'can_use_efficient_attention',return_value=False) as check:
            first = sdpa(q,k,k,enable_gqa=True)
            second = sdpa(q,k,k,enable_gqa=True)
        self.assertEqual(check.call_count, 1)
        self.assertEqual(list(attention.DECISIONS.values()), [False])
        torch.testing.assert_close(first.as_subclass(torch.Tensor), expected, rtol=0, atol=0)
        torch.testing.assert_close(second.as_subclass(torch.Tensor), expected, rtol=0, atol=0)

    def test_refused_kernel_retains_native_instead_of_repeated_attempts(self):
        q,k,expected = self.dispatch_fixture()
        original = F.scaled_dot_product_attention
        calls = []
        def dispatch(*args, **kwargs):
            calls.append(kwargs.get('enable_gqa'))
            if not kwargs.get('enable_gqa'):
                raise RuntimeError('simulated unsupported CUDA kernel')
            return original(*args, **kwargs)
        with patch.object(torch.backends.cuda,'can_use_efficient_attention',return_value=True), \
             patch.object(F,'scaled_dot_product_attention',side_effect=dispatch):
            first = sdpa(q,k,k,enable_gqa=True)
            sdpa(q,k,k,enable_gqa=True)
        self.assertEqual(calls, [False, True, True])
        self.assertEqual(list(attention.DECISIONS.values()), [False])
        torch.testing.assert_close(first.as_subclass(torch.Tensor), expected, rtol=0, atol=0)
