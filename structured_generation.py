from typing import Any
from tinygrad import Tensor, dtypes
from outlines.fsm.guide import CFGGuide, Guide, RegexGuide

class OutlinesLogitsProcessor:
    def process_logits(self, input_ids:Tensor, logits:Tensor) -> Tensor:
        raise NotImplementedError
    def __call__(self, input_ids:Tensor, logits:Tensor) -> Tensor:
        ic(type(input_ids), type(logits), input_ids.shape, logits.shape)
        assert dtypes.is_int(input_ids.dtype), f"input_ids must be integers but have type {input_ids.dtype}"
        assert logits.shape[:-1] == input_ids.shape[:-1], f"logits and input_ids must have the same dimensions except for the last"
        assert logits.ndim in [1, 2], f'logits can only have 1 or 2 dimensions but have {logits.ndim}'
        return self.process_logits(input_ids, logits) if logits.ndim == 2 else self.process_logits(input_ids, logits).squeeze(0)

class GuideLogitsProcessor(OutlinesLogitsProcessor):
    tokenizer: "Tokenizer"
    guide: Guide
    _guide_states: dict[int, Any]
    _seq_start_idx: int|None
    def __init__(self, tokenizer: "Tokenizer", guide: Guide):
        self.tokenizer = tokenizer
        self.guide = guide
        self._guide_states = {hash(tuple([])): self.guide.initial_state}
        self._seq_start_idx = None

    def process_logits(self, input_ids: Tensor, logits: Tensor) -> Tensor:
        """Use the Guide to bias the logits before sampling the next token."""
        if self._seq_start_idx is None:
            self._seq_start_idx = len(input_ids[0])

        sequence_states: list[int] = []  # vector of states corresponding to `input_ids`

        for i in range(input_ids.shape[0]):
            seq_ids = input_ids[i]
            gen_ids = seq_ids[self._seq_start_idx :]
            curr_state_key = hash(tuple(gen_ids.tolist()))

            if curr_state_key not in self._guide_states:
                prev_state = self._guide_states[hash(tuple(gen_ids[:-1].tolist()))]
                curr_state = self.guide.get_next_state(prev_state, gen_ids[-1].item())
                self._guide_states[curr_state_key] = curr_state

            sequence_states.append(self._guide_states[curr_state_key])

        allowed_tokens_batch: list[Tensor] = []
        batch_indices: list[Tensor] = []
        for i, guide_state in enumerate(sequence_states):
            # todo: get_next_instruction should return Tensor directly so don't need to cast torch.Tensor -> tinygrad.Tensor
            allowed_tokens = Tensor(self.guide.get_next_instruction(guide_state).tokens.numpy())
            allowed_tokens_batch.append(allowed_tokens)
            batch_indices.append(Tensor.full_like(allowed_tokens, i))  # Store batch index for each allowed token

        allowed_tokens_concat = allowed_tokens_batch[0].cat(*allowed_tokens_batch[1:]).to(logits.device)
        batch_indices_concat = batch_indices[0].cat(*batch_indices[1:]).to(logits.device)

        mask = Tensor.ones_like(logits, dtype=dtypes.bool).contiguous()
        mask[batch_indices_concat, allowed_tokens_concat] = False
        logits = logits.masked_fill(mask, float("-inf"))

        return logits

class RegexLogitsProcessor(GuideLogitsProcessor):
    """Bias generation based on a regular expression."""

    def __init__(self, regex_string: str, tokenizer: "Tokenizer"):
        guide = RegexGuide.from_regex(regex_string, tokenizer)
        super().__init__(tokenizer=tokenizer, guide=guide)
