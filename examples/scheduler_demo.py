"""Inspect a small schedule: python3 -m examples.scheduler_demo."""
from tiny_vllm.kv_cache import KVCacheManager, Request, SamplingParams
from tiny_vllm.scheduler import Scheduler
from tiny_vllm.scheduler_config import SchedulerConfig


def main():
    scheduler = Scheduler(SchedulerConfig(4, 2, 6), KVCacheManager(32, 4, True))
    for name, length in [('A', 10), ('B', 4), ('C', 7)]:
        scheduler.add(Request(name, list(range(length)), SamplingParams(max_tokens=3)))
    for step in range(100):
        if not scheduler.has_work():
            print('All requests finished; free pages:', scheduler.kv.num_free)
            return
        batch = scheduler.schedule()
        print(f'Round {step + 1}: {sum(item.num_new for item in batch)}/6 token positions')
        for item in batch:
            print(f'  {item.req.request_id}: computed={item.num_computed}, '
                  f'new={item.num_new}, completes_sequence={item.completes_prompt}')
            if item.completes_prompt:
                item.req.token_ids.append(999)  # A stand-in for a sampled token.
                if len(item.req.output_ids) == 3:
                    scheduler.finish(item.req, 'length')
                    print(f'  {item.req.request_id} finished')
    raise RuntimeError('Demo exceeded its scheduling limit')


if __name__ == '__main__':
    main()
