``faninsar.logging``
====================

FanInSAR provides a small logging utility with:

- Colored console output via ``colorlog``
- A custom ``SUCCESS`` level (value 25) between ``INFO`` and ``WARNING``
- ``tqdm``-compatible handler to avoid progress bar corruption
- Smart default log level selection for development vs production

Public API
----------

- ``faninsar.logging.SUCCESS``
- ``faninsar.logging.setup_logger``
- ``faninsar.logging.get_default_log_level``
- ``faninsar.logging.tqdm_handler``
- ``faninsar.logging.stream_handler``

Default Log Level
-----------------

``setup_logger`` uses ``get_default_log_level()`` when ``level`` is not specified, with the following precedence:

1. ``FANINSAR_LOG_LEVEL`` environment variable (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``)
2. ``FANINSAR_DEBUG=1`` (or ``true/yes/on``) -> ``DEBUG``
3. Heuristic development detection (``ENVIRONMENT=development``, interactive shell, test runner) -> ``DEBUG``
4. Fallback -> ``INFO``

Usage
-----

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

Parameters
----------

- ``name`` (alias: ``log_name``): logger name (default: ``"faninsar"``)
- ``file`` (alias: ``log_file``): optional path to a log file
- ``handler``: a handler or list of handlers to attach (default: colored ``stream_handler``)
- ``level``: handler log level; if ``None``, uses ``get_default_log_level()``
- ``propagate``: propagate log records to parent loggers (default: True)
- ``clear_existing``: clear existing handlers before attaching new ones (default: False)

Notes
-----

- The underlying logger level is set to ``DEBUG`` so that handlers can filter appropriately.
- The custom ``SUCCESS`` level is available on all loggers as ``logger.success(...)``.
