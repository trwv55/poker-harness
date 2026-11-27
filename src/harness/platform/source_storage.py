"""Private source objects. Poker contracts never depend on this infrastructure port."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict


class SourceRef(BaseModel):
    model_config = ConfigDict(frozen=True)
    backend: str = 'local'
    key: str
    sha256: str
    size: int
    mime: str

    def encode(self) -> str:
        return self.model_dump_json()


class SourceStorage(Protocol):
    def put(self, owner_id: int, content: bytes, mime: str, operation_key: str) -> SourceRef: ...
    def read(self, owner_id: int, ref: SourceRef, max_bytes: int) -> bytes: ...
    def delete(self, owner_id: int, ref: SourceRef, expected_hash: str) -> None: ...


class LocalSourceStorage:
    """Opaque keys; every read/delete checks the private owner manifest and exact bytes.

    Atomic replace publishes complete objects only. A crash before DB commit leaves an
    orphan, never a partial object. No automatic deletion of permanent sources.
    """
    def __init__(self, root: Path):
        self.root = root.resolve()

    def _path(self, ref: SourceRef) -> Path:
        if ref.backend != 'local' or not re.fullmatch(r'[0-9a-f]{64}', ref.key):
            raise ValueError('unsupported or unsafe source reference')
        path = self.root / 'sources' / ref.key
        if path.resolve() != path.absolute() or path.parent.resolve() != self.root / 'sources':
            raise ValueError('source path escape')
        return path

    def put(self, owner_id: int, content: bytes, mime: str, operation_key: str) -> SourceRef:
        import fcntl
        key = hashlib.sha256(f'{owner_id}:{operation_key}'.encode()).hexdigest()
        ref = SourceRef(key=key, sha256=hashlib.sha256(content).hexdigest(), size=len(content), mime=mime)
        path = self._path(ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.with_suffix('.lock')
        if lock.resolve() != lock.absolute():
            raise ValueError('source lock path escape')
        with lock.open('a') as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            return self._put_locked(owner_id, content, ref, path)

    def _put_locked(self, owner_id: int, content: bytes, ref: SourceRef, path: Path) -> SourceRef:
        manifest = path.with_suffix('.json')
        if manifest.resolve() != manifest.absolute():
            raise ValueError('source manifest path escape')
        metadata = {'owner_id': owner_id, 'ref': ref.model_dump()}
        if manifest.exists():
            if json.loads(manifest.read_text()) != metadata:
                raise ValueError('source operation conflict')
            if path.exists():
                self.read(owner_id, ref, len(content))
                return ref
        # Publish an owned intent BEFORE data, so every crash boundary is discoverable.
        # Deterministic pending files belong to this key/intent, not random temp files.
        for target, data in ((manifest, json.dumps(metadata).encode()), (path, content)):
            temporary = target.with_name(target.name + '.pending')
            if temporary.resolve() != temporary.absolute():
                raise ValueError('source temporary path escape')
            with temporary.open('wb') as file:
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
            temporary.replace(target)
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return ref

    def orphan_candidates(self, owner_id: int, before: float) -> list[SourceRef]:
        """Local recovery capability; caller performs DB reachability/locking.

        Grace applies to unreferenced intents only, never to successful history.
        Legacy data without an owner manifest is retained for operator recovery.
        """
        directory = self.root / 'sources'
        if not directory.exists() or directory.resolve() != directory.absolute():
            return []
        refs: dict[str, SourceRef] = {}
        for manifest in [*directory.glob('*.json'), *directory.glob('*.json.pending')]:
            if manifest.resolve() != manifest.absolute() or manifest.stat().st_mtime >= before:
                continue
            try:
                metadata = json.loads(manifest.read_text())
                ref = SourceRef.model_validate(metadata['ref'])
                if metadata['owner_id'] == owner_id and manifest.name in {ref.key + '.json', ref.key + '.json.pending'}:
                    self._path(ref)
                    refs[ref.key] = ref
            except (ValueError, KeyError):
                continue
        return list(refs.values())

    def read(self, owner_id: int, ref: SourceRef, max_bytes: int) -> bytes:
        path = self._path(ref)
        if ref.size <= 0 or ref.size > max_bytes:
            raise ValueError('source size limit')
        manifest = path.with_suffix('.json')
        if manifest.resolve() != manifest.absolute():
            raise ValueError('source manifest path escape')
        if json.loads(manifest.read_text()) != {'owner_id': owner_id, 'ref': ref.model_dump()}:
            raise ValueError('source owner or metadata mismatch')
        with path.open('rb') as file:
            data = file.read(max_bytes + 1)
        if len(data) != ref.size or hashlib.sha256(data).hexdigest() != ref.sha256:
            raise ValueError('source hash or size mismatch')
        return data

    def delete(self, owner_id: int, ref: SourceRef, expected_hash: str) -> None:
        path = self._path(ref)
        if expected_hash != ref.sha256:
            raise ValueError('source cleanup hash mismatch')
        manifest = path.with_suffix('.json')
        pending_manifest = manifest.with_name(manifest.name + '.pending')
        metadata_path = manifest if manifest.exists() else pending_manifest
        if not path.exists() and not metadata_path.exists():
            return
        if metadata_path.resolve() != metadata_path.absolute() or json.loads(metadata_path.read_text()) != {'owner_id': owner_id, 'ref': ref.model_dump()}:
            raise ValueError('source owner or metadata mismatch')
        if path.exists():
            self.read(owner_id, ref, ref.size)
        for target in (path, manifest, pending_manifest, path.with_name(path.name + '.pending')):
            if target.resolve() != target.absolute():
                raise ValueError('source cleanup path escape')
            target.unlink(missing_ok=True)

    def legacy(self, owner_id: int, path: Path, expected_hash: str) -> SourceRef:
        """Only our old owner's UUID intake files, never an arbitrary host path."""
        root = self.root / 'vision' / str(owner_id)
        if path.parent.resolve() != root or path.resolve() != path.absolute() or not re.fullmatch(r'[0-9a-f]{32}\.image', path.name):
            raise ValueError('unsafe legacy source path')
        with path.open('rb') as file:
            data = file.read(10 * 1024 * 1024 + 1)
        if len(data) > 10 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError('legacy source hash or size mismatch')
        return self.put(owner_id, data, image_mime(data), 'legacy:' + path.name)


def image_mime(data: bytes) -> str:
    if data.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    if data.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    if data.startswith(b'RIFF') and data[8:12] == b'WEBP':
        return 'image/webp'
    raise ValueError('unsupported source MIME')


def source_storage(root: Path, backend: str = 'local') -> SourceStorage:
    if backend != 'local':
        raise ValueError('S3 source adapter is not installed; configure local')
    return LocalSourceStorage(root)
