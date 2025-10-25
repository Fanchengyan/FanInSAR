.. _api_logging:

====================
``faninsar.logging``
====================

FanInSAR provides a small logging utility with:

- Colored console output via ``colorlog``
- A custom ``SUCCESS`` level (value 25) between ``INFO`` and ``WARNING``
- ``tqdm``-compatible handler to avoid progress bar corruption
- Smart default log level selection for development vs production

Quick Start
-----------

.. code-block:: python

    from faninsar.logging import setup_logger, tqdm_handler
    import logging

    # Auto-detect level (recommended)
    logger = setup_logger(__name__)
    logger.info("Starting application")
    logger.success("Initialization finished")

    # Explicit level
    logger = setup_logger(__name__, level=logging.WARNING)

    # With tqdm integration
    logger = setup_logger(__name__, handler=tqdm_handler)

    # Log to a file as well
    logger = setup_logger(__name__, file="app.log", level=logging.INFO)

Default Log Level
-----------------

``setup_logger`` uses ``get_default_log_level()`` when ``level`` is not specified, with the following precedence:

1. ``FANINSAR_LOG_LEVEL`` environment variable (``DEBUG``, ``INFO``, ``SUCCESS``, ``WARNING``, ``ERROR``, ``CRITICAL``)
2. ``FANINSAR_DEBUG=1`` (or ``true/yes/on``) → ``DEBUG``
3. Heuristic development detection (``ENVIRONMENT=development``, interactive shell, test runner) → ``DEBUG``
4. Fallback → ``INFO``

API Reference
-------------

.. currentmodule:: faninsar.logging

Classes
^^^^^^^

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   FaninsarLogger
   TqdmLoggingHandler

Functions
^^^^^^^^^

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   setup_logger
   get_default_log_level

Attributes
^^^^^^^^^^

- ``SUCCESS`` — Custom level between ``INFO`` and ``WARNING``
- ``stream_handler`` — Colored stream handler for stdout
- ``tqdm_handler`` — ``tqdm``-aware stream handler
- ``color_formatter`` — Colored formatter
- ``formatter`` — Plain text formatter
