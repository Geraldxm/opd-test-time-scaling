#!/usr/bin/env python3
"""Fail-closed isolated code-judge executor for Linux.

The parent stages output on the host, while the untrusted process runs under a
dedicated unprivileged host UID in a fresh user/mount/net/pid/ipc/uts namespace
and a chroot containing only explicitly mounted runtime and input paths. Host
outputs are promoted only after a zero exit status.
"""
import argparse
import json
import os
import secrets
import shlex
import shutil
import stat
import subprocess
import tempfile
import time
import signal
import uuid
from pathlib import Path

RUNTIME_DIRS = ("/usr", "/bin", "/lib", "/lib64")
LIMITS = ("--cpu=900", "--as=8589934592", "--nofile=256", "--nproc=256", "--fsize=268435456")


def _pair(item):
    return (Path(item[0]), Path(item[1])) if isinstance(item, (list, tuple)) else (Path(item), Path(item))


MAX_PROMOTE_FILES = 1_000
MAX_PROMOTE_BYTES = 256 * 1024 * 1024


def _scan_tree(root: Path) -> tuple[int, int]:
    count = total = 0
    for current, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(current) / name; mode = os.lstat(path).st_mode; count += 1
            if count > MAX_PROMOTE_FILES: raise RuntimeError("candidate output exceeds file limit")
            if stat.S_ISDIR(mode): continue
            if not stat.S_ISREG(mode): raise RuntimeError(f"unsafe candidate output type: {path}")
            total += os.lstat(path).st_size
            if total > MAX_PROMOTE_BYTES: raise RuntimeError("candidate output exceeds byte limit")
    return count, total


def _check_empty_destination(destination: Path) -> None:
    if not destination.exists() and not destination.is_symlink():
        return
    if not stat.S_ISDIR(os.lstat(destination).st_mode):
        raise RuntimeError(f"sandbox output destination is not a real directory: {destination}")
    if any(destination.iterdir()):
        raise RuntimeError(f"sandbox output destination is not empty: {destination}")


def _copy_promote(source: Path, destination: Path) -> None:
    if not stat.S_ISDIR(os.lstat(source).st_mode): raise RuntimeError("sandbox output stage is not a real directory")
    destination.mkdir(parents=True, exist_ok=True); _check_empty_destination(destination); promotion = destination / f".sandbox-promote-{os.getpid()}-{uuid.uuid4().hex}"; promotion.mkdir()
    copied = files = 0
    try:
        for current, dirs, names in os.walk(source, followlinks=False):
            relative = Path(current).relative_to(source); target_dir = promotion / relative
            for name in dirs:
                mode = os.lstat(Path(current) / name).st_mode; files += 1
                if not stat.S_ISDIR(mode): raise RuntimeError("unsafe candidate directory")
                if files > MAX_PROMOTE_FILES: raise RuntimeError("candidate output exceeds file limit")
                (target_dir / name).mkdir()
            for name in names:
                item = Path(current) / name; before = os.lstat(item); files += 1
                if files > MAX_PROMOTE_FILES or not stat.S_ISREG(before.st_mode): raise RuntimeError(f"unsafe candidate output type: {item}")
                src_fd = os.open(item, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    after = os.fstat(src_fd)
                    if not stat.S_ISREG(after.st_mode) or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino): raise RuntimeError("candidate output changed during promotion")
                    dst_fd = os.open(target_dir / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    try:
                        while chunk := os.read(src_fd, 1024 * 1024):
                            view = memoryview(chunk)
                            while view:
                                written = os.write(dst_fd, view)
                                if written <= 0: raise RuntimeError("candidate output copy failed")
                                view = view[written:]; copied += written
                                if copied > MAX_PROMOTE_BYTES: raise RuntimeError("candidate output exceeds byte limit")
                    finally: os.close(dst_fd)
                finally: os.close(src_fd)
        for item in promotion.iterdir(): os.rename(item, destination / item.name)
    finally: shutil.rmtree(promotion, ignore_errors=True)

def run(command, readonly, writable, pythonpath="", env=None, cwd=None, timeout_seconds=172800, allowed_outputs=None):
    if os.geteuid() != 0:
        raise RuntimeError("hardened judge requires a root Linux launcher")
    if not command or not all(isinstance(arg, str) for arg in command):
        raise ValueError("command must be a non-empty list of strings")
    if cwd is not None and (not isinstance(cwd, str) or not cwd.startswith("/") or "\0" in cwd):
        raise ValueError("cwd must be an absolute in-sandbox path")
    if timeout_seconds < 1:
        raise ValueError("sandbox wall timeout must be positive")
    for required in ("setpriv", "unshare", "mount", "chroot", "prlimit"):
        if not shutil.which(required):
            raise RuntimeError(f"required hardened-sandbox tool unavailable: {required}")
    readonly = [_pair(item) for item in readonly]
    writable = [_pair(item) for item in writable]
    if any(not source.exists() for source, _ in readonly):
        raise RuntimeError("readonly sandbox source is missing")
    for destination, _ in writable:
        _check_empty_destination(destination)
    with tempfile.TemporaryDirectory(prefix="opd-code-judge-", dir="/tmp") as stage:
        stage_path = Path(stage)
        # Use a per-invocation host UID so a shared `nobody` process cannot
        # alter the chroot, writable output stage, or judge handoff.
        sandbox_uid = 1_500_000_000 + secrets.randbelow(250_000_000)
        os.chown(stage_path, sandbox_uid, sandbox_uid)
        os.chmod(stage_path, 0o700)
        root = stage_path / "root"
        work = stage_path / "work"
        root.mkdir(); work.mkdir()
        os.chown(root, sandbox_uid, sandbox_uid)
        os.chown(work, sandbox_uid, sandbox_uid)
        staged_writable = []
        for index, (destination, target) in enumerate(writable):
            staged = work / str(index)
            staged.mkdir()
            os.chown(staged, sandbox_uid, sandbox_uid)
            staged_writable.append((staged, target, destination))
        spec = {
            "root": str(root), "readonly": [[str(a), str(b)] for a, b in readonly],
            "writable": [[str(a), str(b)] for a, b, _ in staged_writable], "command": command,
            "pythonpath": pythonpath, "env": env or {}, "cwd": str(cwd or ""),
            "host_net_inode": os.stat("/proc/self/ns/net").st_ino,
        }
        spec_path = stage_path / "spec.json"
        spec_path.write_text(json.dumps(spec))
        os.chown(spec_path, 0, 0)
        os.chmod(spec_path, 0o444)
        child = [os.environ.get("PYTHON", "/usr/bin/python3"), str(Path(__file__).resolve()), "--child", str(spec_path)]
        launcher = ["setpriv", f"--reuid={sandbox_uid}", f"--regid={sandbox_uid}", "--clear-groups", "--no-new-privs", "--pdeathsig=SIGKILL", "unshare", "--user", "--map-root-user", "--mount", "--net", "--pid", "--ipc", "--uts", "--fork", "--kill-child=SIGKILL", *child]
        process = subprocess.Popen(launcher, start_new_session=True)
        violation = None
        deadline = time.monotonic() + timeout_seconds
        while process.poll() is None:
            try:
                for staged, _, _ in staged_writable: _scan_tree(staged)
            except Exception as exc:
                violation = exc; os.killpg(process.pid, signal.SIGKILL); break
            if time.monotonic() >= deadline:
                violation = TimeoutError(f"judge process exceeded {timeout_seconds}s wall limit")
                os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(0.1)
        result = process.wait()
        if isinstance(violation, TimeoutError):
            raise RuntimeError(str(violation))
        if violation: raise RuntimeError(f"candidate output monitor rejected run: {violation}")
        if result: raise RuntimeError(f"hardened sandbox failed (exit {result})")
        for staged, _, destination in staged_writable:
            _scan_tree(staged)
            if allowed_outputs is not None:
                entries = list(staged.iterdir())
                if {item.name for item in entries} != set(allowed_outputs) or any(
                    not stat.S_ISREG(os.lstat(item).st_mode) for item in entries
                ):
                    raise RuntimeError("sandbox output did not match the allowed file set")
            _copy_promote(staged, destination)


def _mkdir_target(root: Path, target: str, is_dir: bool) -> Path:
    inside = root / target.lstrip("/")
    inside.parent.mkdir(parents=True, exist_ok=True)
    if is_dir:
        inside.mkdir(exist_ok=True)
    else:
        inside.touch(exist_ok=True)
    return inside


def child(spec_path: Path) -> None:
    spec = json.loads(spec_path.read_text())
    if os.stat("/proc/self/ns/net").st_ino == spec["host_net_inode"]:
        raise RuntimeError("network namespace was not isolated")
    root = Path(spec["root"])
    root.mkdir(exist_ok=True)
    subprocess.run(["mount", "--make-rprivate", "/"], check=True)
    for source in RUNTIME_DIRS:
        source_path = Path(source)
        if source_path.exists():
            target = _mkdir_target(root, source, True)
            subprocess.run(["mount", "--bind", source, str(target)], check=True)
            subprocess.run(["mount", "-o", "remount,bind,ro", str(target)], check=True)
    for source, target in [tuple(x) for x in spec["readonly"]] + [("/proc/meminfo", "/proc/meminfo")]:
        source_path = Path(source); target_path = _mkdir_target(root, target, source_path.is_dir())
        subprocess.run(["mount", "--bind", source, str(target_path)], check=True)
        subprocess.run(["mount", "-o", "remount,bind,ro", str(target_path)], check=True)
    for source, target in spec["writable"]:
        target_path = _mkdir_target(root, target, True)
        subprocess.run(["mount", "--bind", source, str(target_path)], check=True)
    tmp = _mkdir_target(root, "/tmp", True)
    subprocess.run(["mount", "-t", "tmpfs", "-o", "size=1G,nosuid,nodev,noexec", "tmpfs", str(tmp)], check=True)
    for device in ("/dev/null", "/dev/zero", "/dev/urandom"):
        target = _mkdir_target(root, device, False)
        subprocess.run(["mount", "--bind", device, str(target)], check=True)
        if device != "/dev/null":
            subprocess.run(["mount", "-o", "remount,bind,ro", str(target)], check=True)
    shm = _mkdir_target(root, "/dev/shm", True)
    subprocess.run(["mount", "-t", "tmpfs", "-o", "size=256M,nosuid,nodev,noexec", "tmpfs", str(shm)], check=True)
    cwd = spec["cwd"] or "/"
    env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "PYTHONNOUSERSITE": "1", "PYTHONPATH": spec["pythonpath"]}
    env.update(spec["env"])
    command = ["chroot", str(root), "/usr/sbin/capsh", "--drop=all", "--caps=", "--inh=", "--noamb", "--", "-c", "exec /usr/bin/env -i \"$@\"", "sandbox-env", "PATH=/usr/bin:/bin", "HOME=/tmp", "PYTHONNOUSERSITE=1", f"PYTHONPATH={env['PYTHONPATH']}"]
    for key, value in spec["env"].items():
        command.append(f"{key}={value}")
    command += ["/usr/bin/prlimit", *LIMITS, "--", "/bin/sh", "-c", f"cd {shlex.quote(cwd)} && exec \"$@\"", "sandbox-command", *spec["command"]]
    os.execvp(command[0], command)


def self_test():
    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp); os.chmod(base, 0o755); readonly = base / "input.txt"; output = base / "output"
        readonly.write_text("ok"); output.mkdir(); sentinel = Path("/var/tmp") / f"opd-host-sentinel-{uuid.uuid4().hex}"
        sentinel.write_text("host")
        try:
            network_probe = '''import socket
if {name for _, name in socket.if_nameindex()} != {"lo"}:
    raise SystemExit("unexpected network interface in isolated namespace")
s = socket.socket()
s.settimeout(0.2)
try:
    s.connect(("1.1.1.1", 53))
except OSError:
    pass
else:
    raise SystemExit("network connection unexpectedly succeeded")
'''
            probe = f'''set -eu
test ! -e {shlex.quote(str(sentinel))}
test ! -S /run/docker.sock
test ! -e /proc/1
if echo nope > /inputs/input 2>/dev/null; then exit 41; fi
mkdir /tmp/x
if mount -t tmpfs tmpfs /tmp/x 2>/dev/null; then exit 42; fi
/usr/sbin/capsh --print > /outputs/caps
/usr/bin/python3 -c {shlex.quote(network_probe)}
/usr/bin/python3 -c 'import resource, sys; sys.exit(0 if resource.getrlimit(resource.RLIMIT_NPROC) == (256, 256) else 1)'
echo ok > /outputs/proof
'''
            command = ["/bin/sh", "-c", probe]
            run(command, [[str(readonly), "/inputs/input"]], [[str(output), "/outputs"]])
        finally:
            sentinel.unlink(missing_ok=True)
        if (output / "proof").read_text().strip() != "ok":
            raise RuntimeError("sandbox proof output is missing")
        caps = (output / "caps").read_text()
        cap_lines = {line.strip() for line in caps.splitlines()}
        if "Current: =" not in cap_lines or "Bounding set =" not in cap_lines:
            raise RuntimeError("sandbox retained process capabilities")
        with tempfile.TemporaryDirectory() as negative:
            target = Path(negative) / "accepted"; target.mkdir(); os.chmod(negative, 0o755)
            host = Path("/var/tmp") / f"opd-promotion-sentinel-{uuid.uuid4().hex}"; host.write_text("host")
            try:
                run(["/bin/sh", "-c", f"ln -s {host} /outputs/host-link; mkfifo /outputs/bad-fifo"], [], [[str(target), "/outputs"]])
            except RuntimeError as exc:
                if not ("output monitor rejected run" in str(exc) or "unsafe candidate output type" in str(exc)):
                    raise RuntimeError(f"output-safety negative probe failed for an unexpected reason: {exc}") from exc
            else:
                raise RuntimeError("unsafe candidate output promotion was accepted")
            finally:
                host.unlink(missing_ok=True)
            if list(target.iterdir()):
                raise RuntimeError("unsafe candidate output reached the host result directory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--self-test", action="store_true"); parser.add_argument("--child", type=Path)
    args = parser.parse_args()
    if args.child: child(args.child)
    elif args.self_test: self_test()
    else: parser.error("use --self-test")
