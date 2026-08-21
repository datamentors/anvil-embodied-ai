"""Apply the configured DDP collective timeout in every training rank."""

from __future__ import annotations

import os
from datetime import timedelta

timeout_minutes = int(os.environ.get("DDP_TIMEOUT_MIN", "30") or "0")
if timeout_minutes > 0:
    try:
        import torch.distributed as distributed
    except ImportError:
        distributed = None

    if distributed is not None:
        try:
            original_init_process_group = distributed.init_process_group
            if not getattr(original_init_process_group, "_anvil_timeout_patched", False):

                def init_process_group(*args, **kwargs):
                    kwargs["timeout"] = timedelta(minutes=timeout_minutes)
                    return original_init_process_group(*args, **kwargs)

                init_process_group._anvil_timeout_patched = True
                init_process_group.__wrapped__ = original_init_process_group
                distributed.init_process_group = init_process_group
                print(
                    "[ddp_shim] torch.distributed collective timeout set to "
                    f"{timeout_minutes} min",
                    flush=True,
                )
        except Exception as error:
            print(
                f"[ddp_shim] WARNING: could not patch init_process_group: {error}",
                flush=True,
            )
