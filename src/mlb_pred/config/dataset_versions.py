"""Schema versions for generated MLB training datasets.

Increment this value whenever a regenerated CSV gains, loses, or redefines a
column. Versioned filenames keep older, checksum-pinned datasets intact.

History
-------
``1_0``
    First one-row-per-game dataset. Carries pregame features plus reproducible
    outcomes for total runs, home margin, total-line error, and spread error.
"""

from __future__ import annotations

TRAINING_DATA_SCHEMA_VERSION = "1_0"
