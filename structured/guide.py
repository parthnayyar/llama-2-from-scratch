from __future__ import annotations
import sys, types
try:
    import torch
    _has_dist = torch.distributed.is_available()
except (ImportError, RuntimeError):
    _has_dist = False

if not _has_dist:
    # Fake minimal API so the type annotations inside device_mesh.py don't crash.
    dist_stub = types.ModuleType("torch.distributed")

    class _FakePGOptions: # pylint: disable=too-few-public-methods
        pass

    class _FakeProcessGroup: # pylint: disable=too-few-public-methods
        Options = _FakePGOptions

    dist_stub.ProcessGroup = _FakeProcessGroup
    dist_stub.is_initialized = lambda: False
    sys.modules["torch.distributed"] = dist_stub
    
import json
from typing import Type, Callable
from pydantic import BaseModel
from outlines_core.fsm.json_schema import build_regex_from_schema
from outlines_core.fsm.guide import Guide, Index, Vocabulary
from dataclasses import dataclass

@dataclass
class LlamaVocabulary(Vocabulary):
    """
    Minimal adapter so Outlines can interrogate your SentencePiece tokenizer.
    """

    _decode: Callable[[list[int]], str]

    @classmethod
    def from_llama_tokenizer(cls, tok) -> LlamaVocabulary:
        return cls(
            vocab_size=tok.vocab_size,
            eos_id=tok.eos_id,
            bos_id=tok.bos_id,
            decode=lambda ids: tok.decode(ids),  # id -> str
            encode=lambda s: tok.encode(s, add_bos=False, add_eos=False)[0]  # str -> id
        )


def build_guide(model_cls: Type[BaseModel], tok) -> Guide:
    """
    Turn a Pydantic class into a token-level FSM Guide that knows
    exactly which token ids are valid next.
    """
    # JSON-schema straight from Pydantic
    schema_dict = model_cls.model_json_schema()

    # JSON-schema ➜ regex string (guaranteed equivalent)  
    regex = build_regex_from_schema(json.dumps(schema_dict))
    # Outlines converts that regex to a DFA under the hood.

    # Build a token-aware index so we can ask “what ids are legal now?”
    vocab = LlamaVocabulary.from_llama_tokenizer(tok)
    index = Index(regex, vocab)

    # Guide = FSM + cursor.
    return Guide(index)

if __name__ == "__main__":
    from pydantic import BaseModel, Field
    from typing import Literal
    from sentencepiece import SentencePieceProcessor

    tokenizer_path = "Llama-2-7b/tokenizer.model"

    class Person(BaseModel):
        name: str
        age: int = Field(ge=0, le=120)
        status: Literal["active", "inactive"]
        tags: None|list[str] = None

    tok = SentencePieceProcessor()
    tok.Load(tokenizer_path)
    guide = build_guide(Person, tok)

    # Start of generation – we haven’t emitted anything yet
    print("Initially allowed ids (should decode to '{'):")
    print([tok.decode([i]) for i in guide.get_tokens()[:10]])
