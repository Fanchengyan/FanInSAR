from __future__ import annotations

from typing import get_type_hints

from faninsar.processing.backends import ArrayReader, ArrayWriter


def test_backend_protocol_annotations_resolve_at_runtime() -> None:
    reader_hints = get_type_hints(ArrayReader.read)
    writer_hints = get_type_hints(ArrayWriter.write)

    assert reader_hints["descriptor"].__name__ == "ArrayDescriptor"
    assert writer_hints["return"].__name__ == "ArrayDescriptor"
