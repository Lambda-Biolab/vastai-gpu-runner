"""SSH and SCP utility functions for cloud GPU instances.

Encodes UTI-project deployment lessons:
- Always redirect stdin with ``</dev/null`` (prevent stdin stealing)
- ConnectTimeout capped at 10s (fast fail on unreachable hosts)
- StrictHostKeyChecking=no (Vast.ai IPs are ephemeral)
"""

from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from vastai_gpu_runner.types import CloudInstance

logger = logging.getLogger(__name__)

# Default SSH options for ephemeral cloud instances
_SSH_OPTS = [
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
]


def ssh_cmd(
    instance: CloudInstance,
    command: str,
    *,
    timeout: int = 30,
) -> tuple[int, str]:
    """Execute a command on a cloud instance via SSH.

    Args:
        instance: Cloud instance with SSH credentials.
        command: Shell command to execute.
        timeout: SSH timeout in seconds.

    Returns:
        Tuple of (return_code, stdout).
    """
    cmd = [
        "ssh",
        *_SSH_OPTS,
        "-o",
        f"ConnectTimeout={min(timeout, 10)}",
        "-p",
        str(instance.ssh_port),
        f"{instance.ssh_user}@{instance.ssh_host}",
        command,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode, result.stdout.strip()
    except subprocess.TimeoutExpired:
        return -1, "SSH timeout"
    except FileNotFoundError:
        return -1, "SSH not available"


def scp_upload(
    instance: CloudInstance,
    local_path: Path,
    remote_path: str,
    *,
    timeout: int = 300,
) -> bool:
    """Upload a file to a cloud instance via SCP.

    Args:
        instance: Cloud instance with SSH credentials.
        local_path: Local file to upload.
        remote_path: Remote destination path.
        timeout: SCP timeout in seconds.

    Returns:
        True if upload succeeded.
    """
    cmd = [
        "scp",
        *_SSH_OPTS,
        "-P",
        str(instance.ssh_port),
        str(local_path),
        f"{instance.ssh_user}@{instance.ssh_host}:{remote_path}",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            logger.error("SCP upload failed for %s: %s", local_path.name, result.stderr[:200])
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("SCP upload timeout for %s", local_path.name)
        return False


def scp_download(
    instance: CloudInstance,
    remote_path: str,
    local_path: Path,
    *,
    timeout: int = 600,
) -> bool:
    """Download a file from a cloud instance via SCP.

    Args:
        instance: Cloud instance with SSH credentials.
        remote_path: Remote file path.
        local_path: Local destination path.
        timeout: SCP timeout in seconds.

    Returns:
        True if download succeeded and file is non-empty.
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "scp",
        *_SSH_OPTS,
        "-P",
        str(instance.ssh_port),
        f"{instance.ssh_user}@{instance.ssh_host}:{remote_path}",
        str(local_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0 and local_path.exists() and local_path.stat().st_size > 0
    except subprocess.TimeoutExpired:
        logger.error("SCP download timeout for %s", remote_path)
        return False
