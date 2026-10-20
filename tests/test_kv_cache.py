import unittest

from tiny_vllm.kv_cache import KVCacheManager, Request, SamplingParams


def request(tokens):
    return Request('test', tokens, SamplingParams())


class KVCacheTests(unittest.TestCase):
    def test_shared_page_stays_live_until_last_release(self):
        mgr = KVCacheManager(5, 4, True)
        a = request(list(range(9)))
        self.assertTrue(mgr.allocate(a, 9))
        b = request(list(range(8)) + [99])
        hits = mgr.find_cached_prefix(b)
        self.assertEqual(hits, a.block_ids[:2])
        self.assertTrue(mgr.allocate(b, 1, hits))
        mgr.free_request(a)
        self.assertTrue(all(mgr.ref_cnt[p] == 1 and p not in mgr.free for p in hits))
        mgr.free_request(b)
        self.assertEqual(mgr.num_free, 4)

    def test_all_cached_prompt_keeps_last_block_for_logits(self):
        mgr = KVCacheManager(5, 4, True)
        a = request(list(range(8)))
        mgr.allocate(a, 8)
        mgr.free_request(a)
        self.assertEqual(len(mgr.find_cached_prefix(request(list(range(8))))), 1)

    def test_oom_does_not_mutate_state(self):
        mgr = KVCacheManager(2, 4, True)
        a = request(list(range(8)))
        before = (list(mgr.free), mgr.ref_cnt[:], dict(mgr.hash_to_block))
        self.assertFalse(mgr.allocate(a, 8))
        self.assertEqual(before, (list(mgr.free), mgr.ref_cnt, mgr.hash_to_block))
        self.assertEqual(a.block_ids, [])

    def test_eviction_removes_old_prefix(self):
        mgr = KVCacheManager(3, 4, True)
        a = request(list(range(8)))
        mgr.allocate(a, 8)
        mgr.free_request(a)
        b = request(list(range(20, 28)))
        mgr.allocate(b, 8)
        self.assertEqual(mgr.find_cached_prefix(request(list(range(9)))), [])

    def test_same_block_after_different_parent_is_not_shared(self):
        mgr = KVCacheManager(8, 4, True)
        a = request(list(range(9)))
        mgr.allocate(a, 9)
        mgr.free_request(a)
        b = request([99] * 4 + list(range(4, 9)))
        self.assertEqual(mgr.find_cached_prefix(b), [])


if __name__ == '__main__':
    unittest.main()
