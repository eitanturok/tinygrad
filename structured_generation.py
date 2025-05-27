from typing import Any, Type
from tinygrad import Tensor, dtypes

from outlines.fsm.json_schema import convert_json_schema_to_str
from outlines_core.fsm.json_schema import build_regex_from_schema

from dataclasses import dataclass
from typing import Any, Callable, Optional, Set, Union

import interegular
from interegular import Pattern
from interegular.fsm import FSM
from outlines_core.fsm.regex import create_fsm_index_tokenizer, make_byte_level_fsm, make_deterministic_fsm

@dataclass(frozen=True)
class Generate:
    tokens: Optional[list[int]|Tensor]

@dataclass(frozen=True)
class Write:
    tokens: list[int]|Tensor

Instruction = Union[Write, Generate]


class RegexGuide:
    """Guide to generate text in the language of a regular expression."""
    initial_state = 0

    def __init__(
        self, tokenizer, regex_string:Optional[str]=None, fsm: Optional[FSM]=None,
        device=None, regex_parser: Callable[[str],Pattern]=interegular.parse_pattern, frozen_tokens:list[str]=[]):
        assert not (regex_string is None and fsm is None), "choose only one of regex_string or fsm"
        if regex_string is not None: assert regex_parser is not None, f"need regex_parser with regex_string"

        if regex_string is not None: fsm = regex_parser(regex_string).to_fsm()
        states_to_token_maps, empty_token_ids, _ = self.create_states_mapping_from_fsm(fsm, tokenizer, frozen_tokens=frozen_tokens)
        self.states_to_token_maps = states_to_token_maps
        self.empty_token_ids = empty_token_ids

        self.eos_tensor = Tensor([tokenizer.eos_token_id], device=device)
        self.initial_state = states_to_token_maps.get_initial_state()

    def create_states_mapping_from_fsm(self, fsm:FSM, tokenizer, frozen_tokens: list[str] = []) -> tuple[Any, Set[int], Set[int]]:
        """Create the variables related to the mapping between states and tokens from an FSM."""
        byte_fsm = make_byte_level_fsm(fsm.reduce(), keep_utf8=True, frozen_tokens=frozen_tokens)
        regex_fsm, _ = make_deterministic_fsm(byte_fsm)
        states_to_token_maps, empty_token_ids = create_fsm_index_tokenizer(regex_fsm, tokenizer)
        return states_to_token_maps, empty_token_ids, regex_fsm.finals

    def get_next_instruction(self, state: int) -> Instruction:
        """Return the next instruction for guided generation.

        The initialization of the guide builds an index which maps FSM states to a
        map from authorized tokens to the state in which the guide needs to move
        if said token is generated. Therefore the authorized tokens at the
        current state are the keys of the map returned by the value of the index
        for current state.

        If the current state is not contained in the end this means that we are
        in a final state of the guide. We only authorize EOS tokens in the final
        state.
        """
        if state == -1: return Write(self.eos_tensor)
        next_tokens_mask = self.states_to_token_maps.get_allowed_tokens(state)
        return Write(self.eos_tensor) if next_tokens_mask is None else Generate(Tensor(next_tokens_mask))

    def get_next_state(self, state: int, token_id: int) -> int:
        if state == -1: return -1
        next_state = self.states_to_token_maps.get_next_state(state, token_id)
        return -1 if next_state is None else next_state

    def is_final_state(self, state: int) -> bool: return state == -1 or self.states_to_token_maps.is_final_state(state)
    def get_index_dict(self): return self.states_to_token_maps.get_transitions()
    def copy(self): return self


class GuideLogitsProcessor:
    tokenizer: "Tokenizer"
    guide: Any
    _guide_states: dict[int, Any]
    _seq_start_idx: int|None
    def __init__(self, tokenizer: "Tokenizer", guide: Any):
        self.tokenizer = tokenizer
        self.guide = guide
        self._guide_states = {hash(tuple([])): self.guide.initial_state}
        self._seq_start_idx = None

    def __call__(self, input_ids:Tensor, logits:Tensor) -> Tensor:
        assert dtypes.is_int(input_ids.dtype), f"input_ids must be integers but {input_ids.dtype=}"
        assert logits.shape[:-1] == input_ids.shape[:-1], f"logits and input_ids must have the same dims except for the last dim: {logits.shape=} {input_ids.shape=}"
        assert logits.ndim in [1, 2], f'logits can only have 1 or 2 dims but {logits.ndim=}'
        return self.process_logits(input_ids, logits) if logits.ndim == 2 else self.process_logits(input_ids.unsqueeze(0), logits.unsqueeze(0)).squeeze(0)

    def process_logits(self, input_ids:Tensor, logits:Tensor) -> Tensor:
        """Use the Guide to bias the logits before sampling the next token."""
        if self._seq_start_idx is None: self._seq_start_idx = len(input_ids[0])
        gen_ids = input_ids[:, self._seq_start_idx:]

        sequence_states: list[int] = []  # vector of states corresponding to `input_ids`
        for i in range(input_ids.shape[0]):
            seq = gen_ids[i]
            curr_key = hash(tuple(seq.tolist()))

            if curr_key not in self._guide_states:
                prev_key = hash(tuple(seq[:-1].tolist()))
                self._guide_states[curr_key] = self.guide.get_next_state(self._guide_states[prev_key], gen_ids[-1].item())

            sequence_states.append(self._guide_states[curr_key])

        data = [(self.guide.get_next_instruction(state).tokens, i) for i, state in enumerate(sequence_states)]
        all_tokens = Tensor.cat(*[tokens for tokens, _ in data]).to(logits.device)
        all_indices = Tensor.cat(*[Tensor.full((len(tokens),), idx, dtype=dtypes.int32) for tokens, idx in data]).to(logits.device)

        # todo: remove contiguous
        mask = Tensor.ones_like(logits, dtype=dtypes.bool).contiguous()
        mask[all_indices, all_tokens] = False
        return logits.masked_fill(mask, float("-inf"))

class RegexLogitsProcessor(GuideLogitsProcessor):
    """Bias generation based on a regular expression."""

    def __init__(self, regex_string: str, tokenizer: "Tokenizer", device=None):
        guide = RegexGuide(tokenizer, regex_string=regex_string, device=device)
        super().__init__(tokenizer=tokenizer, guide=guide)

class JSONLogitsProcessor(GuideLogitsProcessor):
    """Bias generation based on a JSON schema."""
    from pydantic import BaseModel

    def __init__(self, schema: Union[dict, Type[BaseModel], str], tokenizer: "Tokenizer", whitespace_pattern: Optional[str] = None, device=None):
        schema_str = convert_json_schema_to_str(json_schema=schema)
        regex_string = build_regex_from_schema(schema_str, whitespace_pattern)
        guide = RegexGuide(tokenizer, regex_string=regex_string, device=device)
        super().__init__(tokenizer=tokenizer, guide=guide)
