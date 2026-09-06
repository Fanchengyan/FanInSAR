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
    from faninsar.io.storage.artifact_transaction import (
        ArtifactResourceLimits,
        GenerationLease,
        OpenGeneration,
        collect_generations,
        initialize_root,
        open_current_generation,
        stage_generation,
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
    from faninsar.io.storage.provenance import (
        ProcessingEvent,
        ProvenanceRecord,
        SoftwareIdentity,
        StacProvenanceProperties,
    )
    from faninsar.io.storage.scene_store import (
        CoregisteredSceneStore,
        SceneUnit,
        write_scene_unit,
    )
    from faninsar.io.storage.stack_generation import (
        PairGenerationBinding,
        StackResultGeneration,
        UnwrapResultGeneration,
        open_stack_generation,
        open_unwrap_generation,
        publish_stack_generation,
        publish_unwrap_generation,
    )
    from faninsar.io.storage.validation import (
        ManifestValidationError,
        ManifestValidationSummary,
        validate_pipeline_rebuild_manifest,
    )

_EXPORTS: dict[str, tuple[str, str]] = {
    "ArtifactResourceLimits": ("artifact_transaction", "ArtifactResourceLimits"),
    "ChunkAccessLog": ("arrays", "ChunkAccessLog"),
    "ChunkedZarrArrayStore": ("arrays", "ChunkedZarrArrayStore"),
    "CoregisteredSceneStore": ("scene_store", "CoregisteredSceneStore"),
    "GenerationLease": ("artifact_transaction", "GenerationLease"),
    "InterferogramArtifact": ("ifg_store", "InterferogramArtifact"),
    "InterferogramArtifactStore": ("ifg_store", "InterferogramArtifactStore"),
    "InMemoryArrayStore": ("arrays", "InMemoryArrayStore"),
    "IonosphereArtifact": ("ion_store", "IonosphereArtifact"),
    "IonosphereArtifactStore": ("ion_store", "IonosphereArtifactStore"),
    "OpenGeneration": ("artifact_transaction", "OpenGeneration"),
    "PairGenerationBinding": ("stack_generation", "PairGenerationBinding"),
    "SceneUnit": ("scene_store", "SceneUnit"),
    "StackResultGeneration": ("stack_generation", "StackResultGeneration"),
    "UnwrappedArtifact": ("ifg_store", "UnwrappedArtifact"),
    "UnwrapResultGeneration": ("stack_generation", "UnwrapResultGeneration"),
    "collect_generations": ("artifact_transaction", "collect_generations"),
    "initialize_root": ("artifact_transaction", "initialize_root"),
    "open_current_generation": ("artifact_transaction", "open_current_generation"),
    "open_stack_generation": ("stack_generation", "open_stack_generation"),
    "open_unwrap_generation": ("stack_generation", "open_unwrap_generation"),
    "publish_stack_generation": ("stack_generation", "publish_stack_generation"),
    "publish_unwrap_generation": ("stack_generation", "publish_unwrap_generation"),
    "read_ion_correction_artifact": ("ion_store", "read_ion_correction_artifact"),
    "stage_generation": ("artifact_transaction", "stage_generation"),
    "write_ifg_artifact": ("ifg_store", "write_ifg_artifact"),
    "write_ion_correction_artifact": ("ion_store", "write_ion_correction_artifact"),
    "write_ionosphere_artifact": ("ion_store", "write_ionosphere_artifact"),
    "write_scene_unit": ("scene_store", "write_scene_unit"),
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
    "ArtifactResourceLimits",
    "ChunkAccessLog",
    "ChunkedZarrArrayStore",
    "CoregisteredSceneStore",
    "GenerationLease",
    "InMemoryArrayStore",
    "InterferogramArtifact",
    "InterferogramArtifactStore",
    "IonosphereArtifact",
    "IonosphereArtifactStore",
    "ManifestValidationError",
    "ManifestValidationSummary",
    "OpenGeneration",
    "PairGenerationBinding",
    "ProcessingEvent",
    "ProvenanceRecord",
    "SceneUnit",
    "SoftwareIdentity",
    "StacProvenanceProperties",
    "StackResultGeneration",
    "UnwrapResultGeneration",
    "UnwrappedArtifact",
    "collect_generations",
    "initialize_root",
    "open_current_generation",
    "open_stack_generation",
    "open_unwrap_generation",
    "publish_stack_generation",
    "publish_unwrap_generation",
    "read_ion_correction_artifact",
    "stage_generation",
    "validate_pipeline_rebuild_manifest",
    "write_ifg_artifact",
    "write_ion_correction_artifact",
    "write_ionosphere_artifact",
    "write_scene_unit",
    "write_unwrapped_artifact",
]
