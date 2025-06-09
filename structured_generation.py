import inspect, json, warnings
from typing import Any, Callable, Optional, Set, Union, Type
from dataclasses import dataclass

from tinygrad import Tensor, dtypes
from pydantic import BaseModel, create_model

import interegular
from interegular import Pattern
from interegular.fsm import FSM
from outlines_core.fsm.json_schema import build_regex_from_schema
from outlines_core.fsm.regex import create_fsm_index_tokenizer, make_byte_level_fsm, make_deterministic_fsm

class RegexGuide:
    """Create finite state machine guide from a regular expression."""
    initial_state = 0

    def __init__(self, tokenizer, regex_string:Optional[str]=None, fsm: Optional[FSM]=None,
        device=None, regex_parser: Callable[[str],Pattern]=interegular.parse_pattern):
        assert not (fsm is not None and regex_string is not None), "either fsm or regex_str must not be None"
        assert not (fsm is None and regex_string is None), "can not have both fsm or regex_str being None"
        if regex_string is not None: assert regex_parser is not None, f"need regex_parser with regex_string"

        fsm = fsm if regex_string is None else regex_parser(regex_string).to_fsm()
        state_to_tokens, empty_token_ids, _ = self.make_state_to_tokens_mapping(fsm, tokenizer)
        self.state_to_tokens = state_to_tokens
        self.empty_token_ids = empty_token_ids

        self.eos_tensor = Tensor([tokenizer.eos_token_id], device=device)
        self.initial_state = state_to_tokens.get_initial_state()

    def make_state_to_tokens_mapping(self, fsm:FSM, tokenizer) -> tuple[Any, Set[int], Set[int]]:
        """Create a mapping between FSM state and permitted next tokens."""
        byte_fsm = make_byte_level_fsm(fsm.reduce(), keep_utf8=True)
        regex_fsm, _ = make_deterministic_fsm(byte_fsm)
        state_to_tokens, empty_token_ids = create_fsm_index_tokenizer(regex_fsm, tokenizer)
        return state_to_tokens, empty_token_ids, regex_fsm.finals

    def next_tokens_mask(self, state: int) -> Optional[list[int]|Tensor]:
        """Given a FSM state, return the permitted next tokens."""
        if state == -1: return self.eos_tensor
        next_tokens_mask = self.state_to_tokens.get_allowed_tokens(state)
        return self.eos_tensor if next_tokens_mask is None else Tensor(next_tokens_mask)

    def next_state(self, state: int, token_id: int) -> int:
        """Given a FSM state and token id, return the next FSM state."""
        if state == -1: return -1
        next_state = self.state_to_tokens.get_next_state(state, token_id)
        return -1 if next_state is None else next_state

    def is_final_state(self, state: int) -> bool: return state == -1 or self.state_to_tokens.is_final_state(state)
    def get_index_dict(self): return self.state_to_tokens.get_transitions()
    def copy(self): return self

class LogitsProcessor:
    tokenizer: "Tokenizer"
    guide: Any
    _guide_states: dict[int, Any]
    # _seq_start_idx: int|None
    def __init__(self, tokenizer: "Tokenizer", guide: Any):
        self.tokenizer = tokenizer
        self.guide = guide
        self._guide_states = {hash(tuple([])): self.guide.initial_state}
        # self._seq_start_idx = None
        self.gen_ids = None

    def __call__(self, input_ids:Tensor, logits:Tensor) -> Tensor:
        assert dtypes.is_int(input_ids.dtype), f"input_ids must be integers but {input_ids.dtype=}"
        assert logits.shape[:-1] == input_ids.shape[:-1], f"logits and input_ids must have the same dims except for the last dim: {logits.shape=} {input_ids.shape=}"
        assert logits.ndim in [1, 2], f'logits can only have 1 or 2 dims but {logits.ndim=}'
        return self.process_logits(input_ids, logits) if logits.ndim == 2 else self.process_logits(input_ids.unsqueeze(0), logits.unsqueeze(0)).squeeze(0)

    def process_logits(self, input_ids:Tensor, logits:Tensor) -> Tensor:
        """Use the Guide to bias the logits before sampling the next token."""

        # first time, input_ids is the prompt_tokens
        # after that, input_ids always contains the previously generated token with shape (bs, 1)
        # we condition on ALL generated tokens via caching but don't see that explicitly
        bs = input_ids.shape[0]
        if self.gen_ids is None:
            self.gen_ids = [[] for x in range(bs)]
        else:
            for i,x in enumerate(input_ids): self.gen_ids[i].append(x.item())

        # get FSM states for the generated ids
        ic(self._guide_states)
        fsm_states: list[int] = []
        gen_ids = Tensor(self.gen_ids)
        for i in range(bs):
            ic(self._guide_states)
            seq = gen_ids[i]
            curr_key = hash(tuple(seq.tolist()))
            if curr_key not in self._guide_states:
                prev_key = hash(tuple(seq[:-1].tolist()))
                self._guide_states[curr_key] = self.guide.next_state(self._guide_states[prev_key], seq[-1].item())
            fsm_states.append(self._guide_states[curr_key])

        # create the mask
        data = [(self.guide.next_tokens_mask(state), i) for i, state in enumerate(fsm_states)]
        import numpy as np
        ic(data[0][0].shape)
        np.save("data_results/data.npy", data[0][0].numpy())
        all_tokens = Tensor.cat(*[tokens for tokens, _ in data]).to(logits.device)
        all_indices = Tensor.cat(*[Tensor.full((len(tokens),), idx, dtype=dtypes.int32) for tokens, idx in data]).to(logits.device)
        mask = Tensor.ones_like(logits, dtype=dtypes.bool).contiguous()
        mask[all_indices, all_tokens] = False

        # bias the logits
        biased_logits = logits.masked_fill(mask, float("-inf"))
        return biased_logits

# **** RegexLogitsProcessor ****

class RegexLogitsProcessor(LogitsProcessor):
    """Bias generation based on a regular expression."""
    def __init__(self, regex_string: str, tokenizer: "Tokenizer", device=None):
        guide = RegexGuide(tokenizer, regex_string=regex_string, device=device)
        super().__init__(tokenizer=tokenizer, guide=guide)

# **** JSONLogitsProcessor ****

def convert_json_schema_to_str(json_schema:Union[dict, str, Type[BaseModel]]) -> str:
    """Convert a JSON schema to a string."""
    if isinstance(json_schema, dict): return json.dumps(json_schema)
    elif isinstance(json_schema, str): return json_schema
    elif issubclass(json_schema, BaseModel): return json.dumps(json_schema.model_json_schema())
    else: raise ValueError(f"Cannot parse schema {json_schema}. Only supports dictionary, string, or Pydantic class.")

class JSONLogitsProcessor(LogitsProcessor):
    """Bias generation based on a JSON schema."""
    def __init__(self, schema: Union[dict, Type[BaseModel], str], tokenizer:"Tokenizer", whitespace_pattern: Optional[str]=None, device=None):
        schema_str = convert_json_schema_to_str(json_schema=schema)
        regex_string = build_regex_from_schema(schema_str, whitespace_pattern)
        guide = RegexGuide(tokenizer, regex_string=regex_string, device=device)
        super().__init__(tokenizer=tokenizer, guide=guide)
