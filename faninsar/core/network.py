# ruff: noqa: EM101, EM102, FLY002, TRY003, TRY400

"""Mission-neutral identity and asset contracts for Network products.

The Network layer identifies a physical acquisition more precisely than the
date-only :class:`~faninsar.core.pairs.Pair` topology.  These records are
deliberately immutable so that a product cannot change identity after it has
been admitted to an index.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Self

import numpy as np

from faninsar.logging import setup_logger

logger = setup_logger(__name__)


def _require_text(value: object, field_name: str) -> str:
    """Return a non-empty stable identifier or raise ``ValueError``."""
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        logger.error("invalid Network identity field %s=%r", field_name, value)
        raise ValueError(f"{field_name} must be a non-empty string")
    if any(ord(character) < 32 for character in value):
        logger.error("invalid control character in Network field %s", field_name)
        raise ValueError(f"{field_name} must not contain control characters")
    return value


class PhaseConvention(StrEnum):
    """Physical phase orientation declared by a Network product.

    ``PRIMARY_MINUS_SECONDARY`` is FanInSAR's canonical orientation.  The
    reverse orientation is accepted only when an explicit asset transform is
    recorded alongside the product.
    """

    PRIMARY_MINUS_SECONDARY = "primary_minus_secondary"
    SECONDARY_MINUS_PRIMARY = "secondary_minus_primary"


class AssetKind(StrEnum):
    """Kinds of assets whose phase orientation can require normalization."""

    COMPLEX_INTERFEROGRAM = "complex_interferogram"
    WRAPPED_PHASE = "wrapped_phase"
    UNWRAPPED_PHASE = "unwrapped_phase"
    DISPLACEMENT = "displacement"
    LOS = "los"

    # Short aliases are useful to adapters while retaining one serialized
    # vocabulary.  StrEnum aliases do not create additional enum values.
    COMPLEX = "complex_interferogram"
    SCALAR = "displacement"


class AssetTransformOperation(StrEnum):
    """Allowlisted numerical operation applied to normalize an asset."""

    IDENTITY = "identity"
    COMPLEX_CONJUGATE = "complex_conjugate"
    WRAPPED_PHASE_NEGATE_MODULO = "wrapped_phase_negate_modulo"
    UNWRAPPED_PHASE_NEGATE = "unwrapped_phase_negate"
    SCALAR_NEGATE = "scalar_negate"


@dataclass(frozen=True, slots=True)
class AcquisitionKey:
    """Stable identity for one Network acquisition.

    Parameters
    ----------
    acquisition_id : str
        Provider-stable acquisition identifier.  It is intentionally not
        reduced to a date because multiple products can share a date.
    frame, swath, channel, polarization : str
        Dimensions that disambiguate products in a homogeneous Network
        cohort.

    """

    acquisition_id: str
    frame: str
    swath: str
    channel: str
    polarization: str

    def __post_init__(self) -> None:
        """Validate every component of the immutable acquisition identity."""
        for field_name in (
            "acquisition_id",
            "frame",
            "swath",
            "channel",
            "polarization",
        ):
            _require_text(getattr(self, field_name), field_name)

    @property
    def acquisition(self) -> str:
        """Return ``acquisition_id`` using the domain noun used by adapters."""
        return self.acquisition_id

    @property
    def cohort(self) -> tuple[str, str, str, str]:
        """Return dimensions that must remain homogeneous in one analysis."""
        return (self.frame, self.swath, self.channel, self.polarization)

    @property
    def canonical(self) -> str:
        """Return a collision-resistant, deterministic serialized key."""
        return "|".join(
            (
                self.acquisition_id,
                self.frame,
                self.swath,
                self.channel,
                self.polarization,
            )
        )

    def __str__(self) -> str:
        """Return the canonical serialized identity."""
        return self.canonical

    def as_tuple(self) -> tuple[str, str, str, str, str]:
        """Return the key fields in their stable serialization order."""
        return (
            self.acquisition_id,
            self.frame,
            self.swath,
            self.channel,
            self.polarization,
        )


@dataclass(frozen=True, slots=True)
class AssetTransform:
    """Typed, allowlisted normalization for one phase-bearing asset.

    ``AssetTransform.for_convention`` should normally be used to construct a
    transform.  Direct construction remains available for adapters that have
    already applied a transform, but the operation must agree with both the
    asset kind and phase convention.
    """

    asset_kind: AssetKind
    phase_convention: PhaseConvention
    operation: AssetTransformOperation

    def __post_init__(self) -> None:
        """Reject unknown or physically inconsistent transform declarations."""
        try:
            kind = AssetKind(self.asset_kind)
            convention = PhaseConvention(self.phase_convention)
            operation = AssetTransformOperation(self.operation)
        except (TypeError, ValueError) as error:
            logger.error("invalid Network asset transform: %r", self)
            raise ValueError("asset transform uses an unknown enum value") from error
        object.__setattr__(self, "asset_kind", kind)
        object.__setattr__(self, "phase_convention", convention)
        object.__setattr__(self, "operation", operation)

        expected = self._expected_operation(kind, convention)
        if operation is not expected:
            logger.error(
                "asset transform %s is incompatible with %s/%s",
                operation,
                kind,
                convention,
            )
            raise ValueError(
                f"{operation.value} is not valid for {kind.value} "
                f"under {convention.value}"
            )

    @staticmethod
    def _expected_operation(
        asset_kind: AssetKind,
        phase_convention: PhaseConvention,
    ) -> AssetTransformOperation:
        """Return the only valid operation for an asset/convention pair."""
        if phase_convention is PhaseConvention.PRIMARY_MINUS_SECONDARY:
            return AssetTransformOperation.IDENTITY
        if asset_kind is AssetKind.COMPLEX_INTERFEROGRAM:
            return AssetTransformOperation.COMPLEX_CONJUGATE
        if asset_kind is AssetKind.WRAPPED_PHASE:
            return AssetTransformOperation.WRAPPED_PHASE_NEGATE_MODULO
        if asset_kind is AssetKind.UNWRAPPED_PHASE:
            return AssetTransformOperation.UNWRAPPED_PHASE_NEGATE
        # Scalar orientation is not inferable from the phase convention.  The
        # explicit negate operation is the adapter's declaration of intent.
        return AssetTransformOperation.SCALAR_NEGATE

    @classmethod
    def for_convention(
        cls,
        asset_kind: AssetKind | str,
        phase_convention: PhaseConvention | str,
    ) -> Self:
        """Construct the canonical transform for an asset and convention.

        Raises
        ------
        ValueError
            If the asset kind or phase convention is not in the closed table.

        """
        try:
            kind = AssetKind(asset_kind)
            convention = PhaseConvention(phase_convention)
        except (TypeError, ValueError) as error:
            logger.error(
                "unknown phase convention or asset kind: %r/%r",
                phase_convention,
                asset_kind,
            )
            raise ValueError("unknown phase convention or asset kind") from error
        return cls(kind, convention, cls._expected_operation(kind, convention))

    # Explicit alias reads naturally at adapter call sites.
    for_phase_convention = for_convention

    def apply(self, asset: Any) -> Any:
        """Apply the declared normalization while preserving array masks.

        Parameters
        ----------
        asset : array-like
            Numeric complex or phase asset.  Masked arrays are supported and
            retain their mask through the operation.

        Returns
        -------
        array-like
            A new normalized NumPy array (or masked array).

        """
        values = np.asanyarray(asset)
        if not np.issubdtype(values.dtype, np.number):
            logger.error("cannot phase-normalize non-numeric %s asset", self.asset_kind)
            raise TypeError("phase asset must have a numeric dtype")
        if self.operation is AssetTransformOperation.IDENTITY:
            return values.copy()
        if self.operation is AssetTransformOperation.COMPLEX_CONJUGATE:
            if not np.issubdtype(values.dtype, np.complexfloating):
                logger.error("complex conjugation requested for non-complex asset")
                raise TypeError("complex interferogram asset must be complex")
            return np.conjugate(values)
        if self.operation is AssetTransformOperation.WRAPPED_PHASE_NEGATE_MODULO:
            return np.angle(np.exp(-1j * values))
        if self.operation is AssetTransformOperation.UNWRAPPED_PHASE_NEGATE:
            return -values
        if self.operation is AssetTransformOperation.SCALAR_NEGATE:
            return -values
        logger.error("unsupported asset transform operation %s", self.operation)
        raise ValueError(f"unsupported asset transform {self.operation!r}")


@dataclass(frozen=True, slots=True)
class NetworkProductKey:
    """Identity of one pair product in a Network index."""

    primary: AcquisitionKey
    secondary: AcquisitionKey
    product_kind: AssetKind

    def __post_init__(self) -> None:
        """Validate roles, product kind, and homogeneous cohort dimensions."""
        if not isinstance(self.primary, AcquisitionKey) or not isinstance(
            self.secondary, AcquisitionKey
        ):
            logger.error("Network product roles must be AcquisitionKey values")
            raise TypeError("primary and secondary must be AcquisitionKey values")
        if self.primary == self.secondary:
            logger.error("Network product cannot use one acquisition as both roles")
            raise ValueError("primary and secondary acquisitions must differ")
        if self.primary.cohort != self.secondary.cohort:
            logger.error("Network product roles belong to different cohorts")
            raise ValueError("primary and secondary must share cohort dimensions")
        try:
            object.__setattr__(self, "product_kind", AssetKind(self.product_kind))
        except (TypeError, ValueError) as error:
            logger.error("unknown Network product kind %r", self.product_kind)
            raise ValueError("unknown Network product kind") from error

    @property
    def canonical(self) -> str:
        """Return a deterministic product index key."""
        return (
            f"{self.primary.canonical}::{self.secondary.canonical}::"
            f"{self.product_kind.value}"
        )


@dataclass(frozen=True, slots=True)
class NetworkProduct:
    """Immutable product record registered by a Network.

    The record contains metadata and an asset locator; it never accepts a
    Dataset object.  Raster decoding remains the responsibility of Dataset.
    """

    key: NetworkProductKey
    asset_location: str
    geometry_identity: str
    source_software: str
    phase_convention: PhaseConvention
    asset_transform: AssetTransform
    content_digest: str | None = None
    lineage: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate product identity, locator, and transform agreement."""
        if not isinstance(self.key, NetworkProductKey):
            logger.error("Network product key has invalid type")
            raise TypeError("key must be a NetworkProductKey")
        _require_text(self.asset_location, "asset_location")
        _require_text(self.geometry_identity, "geometry_identity")
        _require_text(self.source_software, "source_software")
        try:
            convention = PhaseConvention(self.phase_convention)
        except (TypeError, ValueError) as error:
            logger.error("unknown product phase convention %r", self.phase_convention)
            raise ValueError("unknown product phase convention") from error
        object.__setattr__(self, "phase_convention", convention)
        if not isinstance(self.asset_transform, AssetTransform):
            logger.error("Network product requires a typed asset transform")
            raise TypeError("asset_transform must be an AssetTransform")
        if self.asset_transform.phase_convention is not convention:
            logger.error("product and asset transform phase conventions differ")
            raise ValueError("phase convention must match asset transform")
        if self.content_digest is not None:
            _require_text(self.content_digest, "content_digest")
        if any(not isinstance(item, str) or not item for item in self.lineage):
            logger.error("Network product lineage contains an invalid item")
            raise ValueError("lineage entries must be non-empty strings")

    @property
    def primary(self) -> AcquisitionKey:
        """Return the logical Primary acquisition key."""
        return self.key.primary

    @property
    def secondary(self) -> AcquisitionKey:
        """Return the logical Secondary acquisition key."""
        return self.key.secondary

    @property
    def canonical(self) -> str:
        """Return the immutable identity used for index and lineage binding."""
        transform = self.asset_transform.operation.value
        return (
            f"{self.key.canonical}::{self.geometry_identity}::"
            f"{self.phase_convention.value}::{transform}"
        )

    def normalize_asset(self, asset: Any) -> Any:
        """Normalize an asset according to its declared phase convention."""
        return self.asset_transform.apply(asset)


@dataclass(frozen=True, slots=True)
class NetworkProductIndex:
    """Validated immutable collection of Network product records."""

    products: tuple[NetworkProduct, ...]

    def __post_init__(self) -> None:
        """Reject duplicate products and mixed analysis cohorts."""
        if any(not isinstance(product, NetworkProduct) for product in self.products):
            logger.error("Network product index contains a non-product record")
            raise TypeError("products must contain NetworkProduct values")
        keys = tuple(product.key.canonical for product in self.products)
        if len(keys) != len(set(keys)):
            logger.error("Network product index contains duplicate product keys")
            raise ValueError("Network product keys must be unique")

    @property
    def cohort(self) -> tuple[str, str, str, str, str] | None:
        """Return the one homogeneous cohort, or ``None`` for an empty index."""
        if not self.products:
            return None
        first = self.products[0]
        return (*first.primary.cohort, first.geometry_identity)

    def homogeneous(self) -> Self:
        """Validate and return this index as one analysis cohort.

        Raises
        ------
        ValueError
            If products span frame/swath/channel/polarization or geometry.

        """
        expected = self.cohort
        if expected is not None:
            for product in self.products:
                actual = (*product.primary.cohort, product.geometry_identity)
                if actual != expected:
                    logger.error("Network product index mixes analysis cohorts")
                    raise ValueError(
                        "Network analysis requires one homogeneous cohort"
                    )
        return self


# ``NetworkProductRecord`` is the descriptive name used in manifest-oriented
# code; retaining one class avoids duplicate identity semantics.
NetworkProductRecord = NetworkProduct


__all__ = [
    "AcquisitionKey",
    "AssetKind",
    "AssetTransform",
    "AssetTransformOperation",
    "NetworkProduct",
    "NetworkProductIndex",
    "NetworkProductKey",
    "NetworkProductRecord",
    "PhaseConvention",
]
