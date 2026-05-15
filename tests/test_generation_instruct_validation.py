import pytest
import sys

for _mod in ("core.config", "services.model_manager"):
    if getattr(sys.modules.get(_mod), "__file__", None) is None:
        sys.modules.pop(_mod, None)

from api.routers.generation import (
    _is_instruct_validation_error,
    _run_inference,
    _validate_generation_instruct,
)


def _run_inference_args(*, text="Hello", instruct=None):
    return (
        _DummyModel(), text, None, None, None, instruct, None,
        4, 2.0, 1.0, None, True, True, None, None, None, None,
    )


class _DummyModel:
    sampling_rate = 24000

    def generate(self, **_kwargs):
        raise AssertionError("validation should run before model.generate")


def test_invalid_voice_design_instruct_is_validation_error():
    with pytest.raises(ValueError, match="Unsupported instruct items found"):
        _validate_generation_instruct(
            "Hello",
            "female, young adult, moderate pitch, Speak as a calm narrator",
        )


def test_chinese_dialect_with_supported_english_tags_is_valid():
    _validate_generation_instruct(
        "你好",
        "female, young adult, moderate pitch, 四川话",
    )


def test_run_inference_does_not_report_invalid_instruct_as_oom():
    with pytest.raises(ValueError, match="Unsupported instruct items found") as err:
        _run_inference(*_run_inference_args(
            instruct="female, young adult, moderate pitch, Speak as a calm narrator",
        ))

    assert "ran out of memory" not in str(err.value)


def test_wrapped_instruct_errors_are_classified_as_validation():
    try:
        raise RuntimeError("Unsupported instruct items found in whisper: bad")
    except RuntimeError as err:
        wrapped = RuntimeError("outer wrapper")
        wrapped.__cause__ = err

    assert _is_instruct_validation_error(wrapped) is True
