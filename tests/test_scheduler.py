import unittest

from tiny_vllm.kv_cache import KVCacheManager, Request, SamplingParams
from tiny_vllm.scheduler import Scheduler
from tiny_vllm.scheduler_config import SchedulerConfig


class SchedulerTests(unittest.TestCase):
    def make(self, budget=4, seqs=2, blocks=20, caching=False):
        return Scheduler(SchedulerConfig(4, seqs, budget), KVCacheManager(blocks, 4, caching))

    def req(self, name, length):
        return Request(name, list(range(length)), SamplingParams())

    def test_chunked_prefill_and_decode(self):
        s = self.make()
        a = self.req('a', 6)
        s.add(a)
        first = s.schedule()[0]
        self.assertEqual((first.num_computed, first.num_new), (0, 4))
        self.assertFalse(first.completes_prompt)
        second = s.schedule()[0]
        self.assertEqual((second.num_computed, second.num_new), (4, 2))
        self.assertTrue(second.completes_prompt)
        a.token_ids.append(999)
        self.assertEqual(s.schedule()[0].num_new, 1)

    def test_sequence_limit_and_finish_admits_waiting(self):
        s = self.make(budget=8, seqs=1)
        a, b = self.req('a', 4), self.req('b', 4)
        s.add(a)
        s.add(b)
        self.assertEqual([x.req for x in s.schedule()], [a])
        s.finish(a, 'length')
        self.assertEqual([x.req for x in s.schedule()], [b])
        self.assertTrue(a.finished)
        self.assertEqual(a.finish_reason, 'length')
        s.finish(b, 'length')
        self.assertFalse(s.has_work())
        self.assertEqual(s.kv.num_free, 19)

    def test_budget_shared_by_running_decode_and_waiting_prefill(self):
        s = self.make(budget=4)
        a = self.req('a', 4)
        s.add(a)
        s.schedule()
        a.token_ids.append(999)
        b = self.req('b', 9)
        s.add(b)
        batch = s.schedule()
        self.assertEqual([(x.req.request_id, x.num_new) for x in batch], [('a', 1), ('b', 3)])

    def test_preemption_evicts_newest_and_defers_admission(self):
        s = self.make(budget=8, blocks=3)
        a, b, c = self.req('a', 4), self.req('b', 4), self.req('c', 1)
        s.add(a)
        s.add(b)
        s.schedule()
        a.token_ids.append(111)
        b.token_ids.append(222)
        s.add(c)
        self.assertEqual([x.req for x in s.schedule()], [a])
        self.assertEqual(list(s.waiting), [b, c])
        self.assertEqual(b.num_computed_tokens, 0)
        self.assertEqual(b.block_ids, [])
        self.assertEqual(len(b.token_ids), 5)

    def test_cached_prefix_counts_against_computed_not_budget(self):
        s = self.make(budget=16, caching=True)
        a = self.req('a', 9)
        s.add(a)
        s.schedule()
        s.finish(a, 'length')
        b = self.req('b', 9)
        s.add(b)
        item = s.schedule()[0]
        self.assertEqual((item.num_computed, item.num_new), (8, 1))
        self.assertEqual(b.num_cached_tokens, 8)

    def test_fifo_head_can_block_shorter_request(self):
        s = self.make(budget=8, blocks=2)
        a, b = self.req('a', 8), self.req('b', 1)
        s.add(a)
        s.add(b)
        self.assertEqual(s.schedule(), [])
        self.assertEqual(list(s.waiting), [a, b])


if __name__ == '__main__':
    unittest.main()
