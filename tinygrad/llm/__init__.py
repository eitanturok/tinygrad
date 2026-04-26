__all__ = ["RegexLogitsProcessor", "JSONLogitsProcessor"]

def __getattr__(name: str):
  if name in __all__:
    from .structured_generation import JSONLogitsProcessor, RegexLogitsProcessor
    return {"RegexLogitsProcessor": RegexLogitsProcessor, "JSONLogitsProcessor": JSONLogitsProcessor}[name]
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
