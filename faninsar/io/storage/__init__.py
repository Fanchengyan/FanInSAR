"""Persistent and in-memory storage for FanInSAR products.

The package keeps its public names lazy because storage implementations depend
on processing contracts. Lazy exports prevent importing a storage package from
re-entering the processing package while it is being initialized.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from faninsar.io.storage.arrays import (
        ChunkAccessLog,
        ChunkedZarrArrayStore,
        InMemoryArrayStore,
    )
    from faninsar.io.storage.ifg_store import (
        InterferogramArtifact,
        InterferogramArtifactStore,
        UnwrappedArtifact,
        write_ifg_artifact,
        write_unwrapped_artifact,
    )
    from faninsar.io.storage.ion_store import (
        IonosphereArtifact,
        IonosphereArtifactStore,
        read_ion_correction_artifact,
        write_ion_correction_artifact,
        write_ionosphere_artifact,
    )
    from faninsar.io.storage.product import (
        Product,
        ProductWriter,
        commit_product,
        discover_products,
        open_product,
        verify_product,
    )
    from faninsar.io.storage.provenance import (
        ProcessingEvent,
        ProvenanceRecord,
        SoftwareIdentity,
        StacProvenanceProperties,
    )
    from faninsar.io.storage.scene_store import (
        CoregisteredSceneStore,
        SceneUnit,
        SceneWriter,
        write_scene,
    )
    from faninsar.io.storage.validation import (
        ManifestValidationError,
        ManifestValidationSummary,
        validate_pipeline_rebuild_manifest,
    )

_EXPORTS: dict[str, tuple[str, str]] = {
    "ChunkAccessLog": ("arrays", "ChunkAccessLog"),
    "ChunkedZarrArrayStore": ("arrays", "ChunkedZarrArrayStore"),
    "CoregisteredSceneStore": ("scene_store", "CoregisteredSceneStore"),
    "InterferogramArtifact": ("ifg_store", "InterferogramArtifact"),
    "InterferogramArtifactStore": ("ifg_store", "InterferogramArtifactStore"),
    "InMemoryArrayStore": ("arrays", "InMemoryArrayStore"),
    "IonosphereArtifact": ("ion_store", "IonosphereArtifact"),
    "IonosphereArtifactStore": ("ion_store", "IonosphereArtifactStore"),
    "SceneUnit": ("scene_store", "SceneUnit"),
    "UnwrappedArtifact": ("ifg_store", "UnwrappedArtifact"),
    "read_ion_correction_artifact": ("ion_store", "read_ion_correction_artifact"),
    "write_ifg_artifact": ("ifg_store", "write_ifg_artifact"),
    "write_ion_correction_artifact": ("ion_store", "write_ion_correction_artifact"),
    "write_ionosphere_artifact": ("ion_store", "write_ionosphere_artifact"),
    "write_scene": ("scene_store", "write_scene"),
    "SceneWriter": ("scene_store", "SceneWriter"),
    "Product": ("product", "Product"),
    "ProductWriter": ("product", "ProductWriter"),
    "commit_product": ("product", "commit_product"),
    "discover_products": ("product", "discover_products"),
    "open_product": ("product", "open_product"),
    "verify_product": ("product", "verify_product"),
    "write_unwrapped_artifact": ("ifg_store", "write_unwrapped_artifact"),
    "ProcessingEvent": ("provenance", "ProcessingEvent"),
    "ProvenanceRecord": ("provenance", "ProvenanceRecord"),
    "SoftwareIdentity": ("provenance", "SoftwareIdentity"),
    "StacProvenanceProperties": ("provenance", "StacProvenanceProperties"),
    "ManifestValidationError": ("validation", "ManifestValidationError"),
    "ManifestValidationSummary": ("validation", "ManifestValidationSummary"),
    "validate_pipeline_rebuild_manifest": (
        "validation",
        "validate_pipeline_rebuild_manifest",
    ),
}


def __getattr__(name: str) -> Any:
    """Load a storage export on first access."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "ChunkAccessLog",
    "ChunkedZarrArrayStore",
    "CoregisteredSceneStore",
    "InMemoryArrayStore",
    "InterferogramArtifact",
    "InterferogramArtifactStore",
    "IonosphereArtifact",
    "IonosphereArtifactStore",
    "ManifestValidationError",
    "ManifestValidationSummary",
    "ProcessingEvent",
    "Product",
    "ProductWriter",
    "ProvenanceRecord",
    "SceneUnit",
    "SceneWriter",
    "SoftwareIdentity",
    "StacProvenanceProperties",
    "UnwrappedArtifact",
    "commit_product",
    "discover_products",
    "open_product",
    "read_ion_correction_artifact",
    "validate_pipeline_rebuild_manifest",
    "verify_product",
    "write_ifg_artifact",
    "write_ion_correction_artifact",
    "write_ionosphere_artifact",
    "write_scene",
    "write_unwrapped_artifact",
]
