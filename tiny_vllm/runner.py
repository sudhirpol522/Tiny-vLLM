"""Upstream eager ModelRunner adapted to CPU/MPS; see UPSTREAM.md."""
import numpy as np
import torch

from tiny_vllm import model as model_module
from tiny_vllm.attention import AttnMetadata, torch_ref_attention
from tiny_vllm.loading import load_model
from tiny_vllm.sampling import sample
from tiny_vllm.scheduler import ScheduledRequest


class ModelRunner:
    """Pack scheduled tokens, write per-layer K/V, run a forward, and sample."""

    def __init__(self, cfg):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.model = load_model(cfg.model, cfg.dtype, cfg.max_model_len, cfg.device)
        mc = self.model.cfg
        self.H, self.Hkv = mc.num_attention_heads, mc.num_key_value_heads
        self.D = getattr(mc, 'head_dim', None) or mc.hidden_size // self.H
        self.num_layers = mc.num_hidden_layers
        model_module.ATTN_FN = torch_ref_attention
        self.max_blocks_per_seq = -(-cfg.max_model_len // cfg.block_size)
        n, seqs = cfg.max_num_batched_tokens, cfg.max_num_seqs
        z = lambda *shape, dt=torch.int32: torch.zeros(*shape, dtype=dt, device=cfg.device)
        self.input_ids, self.positions = z(n, dt=torch.int64), z(n, dt=torch.int64)
        self.slot_mapping = z(n, dt=torch.int64)
        self.query_start_loc, self.cu_seqlens_k = z(seqs + 1), z(seqs + 1)
        self.block_table = z(seqs, self.max_blocks_per_seq)

    @torch.inference_mode()
    def allocate_kv_cache(self):
        # Explicit capacity replaces CUDA memory profiling on the Mac.
        for layer in self.model.layers:
            layer.self_attn.kv_cache = torch.zeros(
                2, self.cfg.num_blocks, self.cfg.block_size, self.Hkv, self.D,
                dtype=self.cfg.dtype, device=self.cfg.device)
        return self.cfg.num_blocks

    def _prepare(self, batch: list[ScheduledRequest], pad_to: int = 0, for_graph: bool = False):
        """
        
        Pack this step into the persistent buffers the model / graphs read.

        Builds packed input_ids, positions, slot_mapping, query_start_loc,
        cu_seqlens_k, block_table, and AttnMetadata. Addresses of those
        buffers never change, which is what lets a CUDA graph replay.

        A decode batch is "how many live requests each produce one new
        token this step":

            A already has 40 tokens -> run token 41
            B already has 12 tokens -> run token 13
            C already has  7 tokens -> run token  8

            num_new = 1 for each
            T = 3
            q = [qA, qB, qC]          packed
            kv_len = 41, 13, 8        each attends over its own prefix

        Next step, if all three still run, T=3 again. A graph captured at
        batch size 4 serves this by adding one dummy row.

        pad_to > len(batch):
            Decode graphs are captured at a fixed batch size (T == B == pad_to,
            q_len == 1). A live decode batch smaller than that is padded with
            dummy 1-token requests: token 0, position 0, slot 0 (the null
            page). Their attention is ignored, logit_rows only lists real
            requests. Without this pad, T would change every step and the
            captured graph would be the wrong shape.

        Prefill / mixed batches (some num_new > 1) call this with pad_to=0
        and for_graph=False. T is the real token count and the forward
        runs eager. Graphs are decode-only because only then is T a
        function of batch size alone. One prefill chunk of 20 next to two
        decodes would make T=22 and miss the decode graph.
        """
        bs = self.cfg.block_size

        # packed streams we will copy onto the gpu buffers
        ids, pos, slots, logit_rows = [], [], [], []
        qsl, seq_lens = [0], [] # qsl starts at 0; we append the rest

        # page table for real + dummy rows
        bt = np.zeros((max(len(batch), pad_to), self.max_blocks_per_seq), dtype=np.int32)

        # prepare batches (each sample in the batch is one of the ScheduledRequest)
        for i, s in enumerate(batch):
            start, n = s.num_computed, s.num_new          # already-have-KV, this-step queries
            p = np.arange(start, start + n)               # absolute positions = q_pos
            blocks = np.asarray(s.req.block_ids)          # what are these requests block_ids

            # pack together each batch into a single stream
            ids.extend(s.req.token_ids[start:start + n])  # packed input ids
            # position embeds restart from 0 for every sample
            pos.extend(p.tolist())                        # RoPE / causal positions

            # slot = page * bs + offset, destination of index_copy_ for each new token
            slots.extend((blocks[p // bs] * bs + p % bs).tolist())

            qsl.append(qsl[-1] + n)                       # request i owns q[qsl[i]:qsl[i+1]]
            seq_lens.append(start + n)                    # kv_len after this step's write
            bt[i, : len(blocks)] = blocks                 # page-table row
            logit_rows.append(qsl[-1] - 1)                # last packed row -> sample logits here

        # Dummy decode reqeuests so T and B match the captured graph, 
        # slot 0 writes into the null page, outputs are never sampled
        # when doing graph capture, we dont pass in data, just padding
        # so all our q_len = 1 for all cases, which is the case of our
        # deocding anyway, so our graph capture will optimize the decoding 
        # cases for us
        for _ in range(len(batch), pad_to):
            ids.append(0), pos.append(0), slots.append(0)
            qsl.append(qsl[-1] + 1), seq_lens.append(1)

        T, B = len(ids), len(seq_lens)

        # persistent buffers: fixed addresses for CUDA graphs
        self.input_ids[:T].copy_(torch.tensor(ids))
        self.positions[:T].copy_(torch.tensor(pos))
        self.slot_mapping[:T].copy_(torch.tensor(slots))
        self.query_start_loc[: B + 1].copy_(torch.tensor(qsl))
        self.cu_seqlens_k[: B + 1].copy_(torch.tensor([0] + np.cumsum(seq_lens).tolist()))
        self.block_table[:B].copy_(torch.from_numpy(bt))

        # build the attn metadata
        max_seq = self.cfg.max_model_len if for_graph else max(seq_lens)

        meta = AttnMetadata(
            self.slot_mapping[:T],
            self.query_start_loc[: B + 1],
            self.cu_seqlens_k[: B + 1],
            self.block_table[:B],
            B,
            max(s.num_new for s in batch) if batch else 1,  # max_query_len
            max_seq,
            qsl, seq_lens,                                  # host copies for torch_ref
        )
        return T, meta, logit_rows

    @torch.inference_mode()
    def run(self, batch: list[ScheduledRequest]) -> tuple[list[int], torch.Tensor]:
        if not batch:
            raise ValueError('Cannot run an empty batch')
        total, meta, rows = self._prepare(batch)
        previous = model_module.ATTN_META
        try:
            model_module.ATTN_META = meta
            hidden = self.model(self.input_ids[:total], self.positions[:total])
            logits = self.model.logits(hidden[rows])
        finally:
            model_module.ATTN_META = previous
        return sample(logits, [item.req.params for item in batch]), logits
