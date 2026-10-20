"""Run with python3 -m examples.kv_demo."""
from tiny_vllm.kv_cache import KVCacheManager, Request, SamplingParams

def kv_cache_demo():
    B = 4
    mgr = KVCacheManager(num_blocks=8, block_size=B, enable_caching=True)

    def cached_pages():
        return [b for b, h in enumerate(mgr.block_hash) if h is not None]

    def dump(label, *reqs):
        print(f"\n=== {label} ===")
        print(f"  free (LRU front): {list(mgr.free)}")
        print(f"  ref_cnt:          {mgr.ref_cnt}")
        print(f"  hashed pages:     {cached_pages()}")
        for r in reqs:
            print(f"  {r.request_id}: {r.num_tokens} tok, computed={r.num_computed_tokens}, "
                  f"pages={r.block_ids}, hashes={len(r.block_hashes)}")

    dump("fresh (page 0 reserved)")

    # 1. chunked prefill
    a = Request("A", [10, 11, 12, 13, 14, 15], SamplingParams(max_tokens=4))
    print("\nA: 6-token prompt, budget 4. admit -> lookup once.")
    print("  hits", mgr.find_cached_prefix(a), mgr.allocate(a, 4))
    a.num_computed_tokens = 4
    dump("A chunk 1, page 1 full, hashed", a)

    print("\nA still live: rest of prompt. no lookup.")
    print("  allocate +2", mgr.allocate(a, 2))
    a.num_computed_tokens = 6
    dump("A prompt done, page 2 half full, not hashed", a)

    # 2. decode
    print("\nA decodes two tokens into the slots in page 2.")
    for t in (16, 17):
        a.token_ids.append(t)
        mgr.allocate(a, 1)
        a.num_computed_tokens += 1
    dump("A = 8 tokens, page 2 now hashed", a)

    # 3. B shares first block only
    b = Request("B", [10, 11, 12, 13, 77, 88], SamplingParams(max_tokens=4))
    hits = mgr.find_cached_prefix(b)
    print("\nB admit: same first 4 tokens.")
    print("  hits", hits, mgr.allocate(b, 2, cached=hits))
    b.num_computed_tokens = len(hits) * B + 2
    dump("B attached page 1, new page for 77,88", a, b)
    print(f"  page {hits[0]} ref_cnt={mgr.ref_cnt[hits[0]]} (A and B)")

    # 4. why max_blocks uses num_tokens-1
    c = Request("C", a.token_ids[:8], SamplingParams(max_tokens=4))
    print("\nC admit: exactly A's 8 tokens (2 full hashed blocks).")
    print("  hits", mgr.find_cached_prefix(c), ", one block, not two.")
    print("  (num_tokens-1)//B leaves one token to run; cache has K/V, not logits.")

    # 5. free A
    print("\nA finishes. tail page freed first.")
    mgr.free_request(a)
    dump("A gone, page 2 free but still hashed; page 1 held by B", b)

    # 6. eviction
    d = Request("D", list(range(200, 220)), SamplingParams(max_tokens=4))
    h2 = a.block_hashes[1]
    print("\nD needs 5 pages; free =", list(mgr.free))
    print("  A's 2nd block lives on page", mgr.hash_to_block.get(h2), "(front of free)")
    print("  allocate D +20", mgr.allocate(d, 20))
    d.num_computed_tokens = 20
    dump("D took every free page", b, d)
    print("  A's 2nd block now at", mgr.hash_to_block.get(h2), "(evicted with page 2)")
    print("  page 1 lives: B still holds it.")

    # 7. OOM on pages
    e = Request("E", [1, 2, 3, 4], SamplingParams(max_tokens=4))
    print("\nE: no free pages.")
    print("  allocate E +4", mgr.allocate(e, 4), ", no mutation")

if __name__ == "__main__":
    kv_cache_demo()
