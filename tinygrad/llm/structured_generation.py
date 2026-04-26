import json
from typing import Any, Callable, Optional, Set, Union, Type

import numpy as np
from outlines_core import Index, Vocabulary
from outlines_core.json_schema import build_regex_from_schema
try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import resolve

class RegexGuide:
    """Create finite state machine guide from a regular expression."""
    initial_state = 0

    def __init__(self, regex_string: str, tokenizer, eos_token_id: int, device=None):
        import re as _re
        regex_string = _re.sub(r'^\^', '', _re.sub(r'\$$', '', regex_string))
        vocab_map: dict[str, list[int]] = {}
        for tid, token_bytes in tokenizer._tok2bytes.items():
            if tid == eos_token_id: continue
            token_str = token_bytes.decode('utf-8', errors='replace')
            vocab_map.setdefault(token_str, []).append(tid)
        vocabulary = Vocabulary(eos_token_id, vocab_map)
        self.index = Index(regex_string, vocabulary)
        self.eos_token_id = eos_token_id
        self.eos_tensor = Tensor([eos_token_id], device=device)
        self.initial_state = self.index.get_initial_state()

    def next_tokens_mask(self, state: int) -> Optional[list[int]|Tensor]:
        """Given a FSM state, return the permitted next tokens."""
        if state == -1: return self.eos_tensor
        allowed = self.index.get_allowed_tokens(state)
        return self.eos_tensor if allowed is None else Tensor(allowed)

    def next_state(self, state: int, token_id: int) -> int:
        """Given a FSM state and token id, return the next FSM state."""
        if state == -1: return -1
        next_state = self.index.get_next_state(state, token_id)
        return -1 if next_state is None else next_state

    def is_final_state(self, state: int) -> bool: return state == -1 or self.index.is_final_state(state)
    def get_index_dict(self): return self.index.get_transitions()
    def copy(self): return self

class LogitsProcessor:
    def __init__(self, guide: Any, tokenizer=None):
        self.guide = guide
        self.tokenizer = tokenizer
        self._guide_states: dict[tuple[int, ...], Any] = {(): self.guide.initial_state}
        self._gen_start: int = -1  # -1 = not yet set; set on first rollout call

    def _decode(self, token_id: int) -> str:
        if self.tokenizer is None: return str(token_id)
        try: return repr(self.tokenizer._tok2bytes[token_id].decode('utf-8', errors='replace'))
        except Exception: return str(token_id)

    def _walk_to_state(self, seq: list[int]) -> int:
        """Walk the FSM from the initial state through seq, caching all intermediate states."""
        for end in range(len(seq) + 1):
            k = tuple(seq[:end])
            if k not in self._guide_states:
                prev_k = tuple(seq[:end-1])
                self._guide_states[k] = self.guide.next_state(self._guide_states[prev_k], seq[end-1])
        return self._guide_states[tuple(seq)]

    def __call__(self, input_ids:Tensor, logits:Tensor, seq_tokens: list[int]) -> Tensor:
        assert dtypes.is_int(input_ids.dtype), f"input_ids must be integers but {input_ids.dtype=}"
        assert logits.shape[:-1] == input_ids.shape[:-1], f"logits and input_ids must have the same dims except for the last dim: {logits.shape=} {input_ids.shape=}"
        assert logits.ndim in [1, 2], f'logits can only have 1 or 2 dims but {logits.ndim=}'
        return self.process_logits(input_ids, logits, seq_tokens) if logits.ndim == 2 else self.process_logits(input_ids.unsqueeze(0), logits.unsqueeze(0), seq_tokens).squeeze(0)

    def process_logits(self, input_ids:Tensor, logits:Tensor, seq_tokens: list[int]) -> Tensor:
        """Use the Guide to bias the logits before sampling the next token.

        In HF's API input_ids is the full sequence, but tinygrad uses KV caching so
        tokens during rollout is just the last single token. We reconstruct the full
        generated sequence from the Python-side token list maintained by generate().
        gen_start marks where the prompt ends so we can slice out just generated tokens.
        """
        bs = input_ids.shape[0]
        is_prefill = resolve(input_ids.shape[1] != 1)
        # On prefill: reset FSM, then wait for the first rollout to lock in the prompt length.
        # On first rollout (gen_start not yet set): freeze gen_start = current sequence length.
        # If gen_start goes stale (e.g. second query with warm KV cache skips prefill entirely):
        #   detected by gen_start > len(seq_tokens), reset everything.
        if is_prefill:
            self._gen_start = -1
            self._guide_states = {(): self.guide.initial_state}
        elif self._gen_start == -1:
            self._gen_start = len(seq_tokens) - 1
        elif self._gen_start > len(seq_tokens):
            # Stale: new query with warm cache, no prefill was called.
            self._gen_start = len(seq_tokens) - 1
            self._guide_states = {(): self.guide.initial_state}

        print(f"[DBG] is_prefill={is_prefill} _gen_start={self._gen_start} len(seq_tokens)={len(seq_tokens)}", flush=True)
        print(f"[DBG] seq_tokens[-5:]={[(t, self._decode(t)) for t in seq_tokens[-5:]]}", flush=True)
        print(f"[DBG] _guide_states keys (num={len(self._guide_states)}): {list(self._guide_states.values())}", flush=True)

        # During rollout, seq_tokens already includes the token currently being rolled through the model,
        # so the FSM state here should correspond to the already-generated output prefix.
        gen_seq: list[int] = [] if is_prefill else list(seq_tokens[self._gen_start:])
        curr_key = tuple(gen_seq)
        print(f"[DBG] gen_seq={[(t, self._decode(t)) for t in gen_seq]} curr_key_in_states={curr_key in self._guide_states}", flush=True)
        if curr_key not in self._guide_states:
            print(f"[DBG]   walking FSM from scratch for gen_seq len={len(gen_seq)}", flush=True)
            self._walk_to_state(gen_seq)
        fsm_state = self._guide_states[curr_key]
        print(f"[DBG] fsm_state={fsm_state}", flush=True)

        fsm_states: list[int] = [fsm_state] * bs
        print(f"[DBG] fsm_states={fsm_states}", flush=True)
        vocab_size = logits.shape[-1]
        bias_np = np.full((bs, vocab_size), float('-inf'), dtype=np.float32)
        for i, state in enumerate(fsm_states):
            if state == -1:
                # Final state: only EOS is allowed
                bias_np[i, self.guide.eos_token_id] = 0.0
            else:
                allowed = self.guide.index.get_allowed_tokens(state)
                if allowed is not None:
                    bias_np[i, allowed] = 0.0
        return logits + Tensor(bias_np, device=logits.device)

class RegexLogitsProcessor(LogitsProcessor):
    """Bias generation based on a regular expression."""
    def __init__(self, regex_string: str, tokenizer: "Tokenizer", eos_token_id: int = 0, device=None):
        guide = RegexGuide(regex_string, tokenizer, eos_token_id=eos_token_id, device=device)
        super().__init__(guide=guide, tokenizer=tokenizer)

def convert_json_schema_to_str(json_schema:Union[dict, str, Type[Any]]) -> str:
    """Convert a JSON schema to a string."""
    if isinstance(json_schema, dict): return json.dumps(json_schema)
    elif isinstance(json_schema, str): return json_schema
    elif BaseModel is not None and isinstance(json_schema, type) and issubclass(json_schema, BaseModel): return json.dumps(json_schema.model_json_schema())
    else: raise ValueError(f"Cannot parse schema {json_schema}. Only supports dictionary, string, or Pydantic class.")

class JSONLogitsProcessor(LogitsProcessor):
    """Bias generation based on a JSON schema."""
    def __init__(self, schema: Union[dict, Type[Any], str], tokenizer:"Tokenizer", eos_token_id: int = 0, whitespace_pattern: Optional[str]=None, device=None):
        schema_str = convert_json_schema_to_str(json_schema=schema)
        regex_string = build_regex_from_schema(schema_str, whitespace_pattern)
        guide = RegexGuide(regex_string, tokenizer, eos_token_id=eos_token_id, device=device)
        super().__init__(tokenizer=tokenizer, guide=guide)
