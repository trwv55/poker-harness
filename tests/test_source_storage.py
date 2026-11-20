"""Reusable local/fake source contract: no SDK or network."""
import hashlib

import pytest

from harness.platform.source_storage import LocalSourceStorage, SourceRef, source_storage


class FakeStorage:
    def __init__(self):
        self.objects = {}

    def put(self, owner_id, content, mime, operation_key):
        key = hashlib.sha256(f'{owner_id}:{operation_key}'.encode()).hexdigest()
        ref = SourceRef(backend='fake', key=key, sha256=hashlib.sha256(content).hexdigest(), size=len(content), mime=mime)
        if key in self.objects and self.objects[key] != (owner_id, ref, content):
            raise ValueError('operation conflict')
        self.objects[key] = (owner_id, ref, content)
        return ref

    def read(self, owner_id, ref, max_bytes):
        owner, expected, data = self.objects[ref.key]
        if owner != owner_id or ref != expected or len(data) > max_bytes:
            raise ValueError('owner/metadata/limit')
        return data

    def delete(self, owner_id, ref, expected_hash):
        if expected_hash != ref.sha256:
            raise ValueError('hash')
        if ref.key in self.objects:
            self.read(owner_id, ref, ref.size)
            del self.objects[ref.key]


@pytest.fixture(params=['local', 'fake'])
def storage(request, tmp_path):
    return LocalSourceStorage(tmp_path) if request.param == 'local' else FakeStorage()


def test_source_contract(storage):
    data = b'synthetic private original bytes'
    ref = storage.put(71, data, 'image/png', 'operation-one')
    assert ref == storage.put(71, data, 'image/png', 'operation-one')
    assert '71' not in ref.key and '://' not in ref.key
    assert storage.read(71, SourceRef.model_validate_json(ref.encode()), len(data)) == data
    for owner, target, limit in [(72, ref, len(data)), (71, ref, 1),
        (71, ref.model_copy(update={'sha256': '0'*64}), len(data))]:
        with pytest.raises(ValueError):
            storage.read(owner, target, limit)
    with pytest.raises(ValueError):
        storage.put(71, b'changed', 'image/png', 'operation-one')
    with pytest.raises(ValueError):
        storage.delete(72, ref, ref.sha256)
    storage.delete(71, ref, ref.sha256)
    storage.delete(71, ref, ref.sha256)


def test_local_legacy_and_escape(tmp_path):
    store = LocalSourceStorage(tmp_path)
    directory = tmp_path/'vision'/'71'
    directory.mkdir(parents=True)
    image = b'\x89PNG\r\n\x1a\nsynthetic'
    path = directory/('a'*32+'.image')
    path.write_bytes(image)
    sha = hashlib.sha256(image).hexdigest()
    ref = store.legacy(71, path, sha)
    assert store.read(71, ref, 1024) == image
    for owner, target in [(72, path), (71, tmp_path/'secret'), (71, directory/'../../secret')]:
        with pytest.raises(ValueError):
            store.legacy(owner, target, sha)
    outside = tmp_path/'outside'
    outside.write_bytes(image)
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError):
        store.legacy(71, path, sha)
    source = tmp_path/'sources'/ref.key
    source.unlink()
    source.symlink_to(outside)
    with pytest.raises(ValueError):
        store.read(71, ref, 1024)
    with pytest.raises(ValueError):
        source_storage(tmp_path, 's3')


def test_local_concurrent_operation_conflict_and_crash_recovery(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    store=LocalSourceStorage(tmp_path)
    def write(data):
        try:
            return store.put(19,data,'image/png','same-operation')
        except ValueError:
            return None
    with ThreadPoolExecutor(2) as pool:
        refs=list(pool.map(write,[b'first',b'second']))
    assert sum(ref is not None for ref in refs)==1
    ref=next(ref for ref in refs if ref is not None)
    assert store.read(19,ref,20) in [b'first',b'second']
    # A durable object without DB rows is reusable; delete crash after unlink
    # can be retried from its private manifest without touching another object.
    (tmp_path/'sources'/ref.key).unlink()
    store.delete(19,ref,ref.sha256)
    store.delete(19,ref,ref.sha256)


def test_local_volume_backup_restores_portable_references(tmp_path):
    import shutil
    original=LocalSourceStorage(tmp_path/'original')
    ref=original.put(19,b'original source','image/jpeg','source')
    shutil.copytree(tmp_path/'original',tmp_path/'restore')
    restored=LocalSourceStorage(tmp_path/'restore')
    assert restored.read(19,SourceRef.model_validate_json(ref.encode()),1024)==b'original source'


def test_put_does_not_follow_manifest_symlink(tmp_path):
    store=LocalSourceStorage(tmp_path)
    ref=store.put(19,b'original','image/png','operation')
    manifest=tmp_path/'sources'/(ref.key+'.json')
    outside=tmp_path/'unrelated.json'
    outside.write_text(manifest.read_text())
    manifest.unlink();manifest.symlink_to(outside)
    with pytest.raises(ValueError,match='manifest path escape'):
        store.put(19,b'original','image/png','operation')
    assert outside.is_file() and not outside.is_symlink()


def test_crash_after_intent_before_data_is_discoverable_and_retriable(tmp_path,monkeypatch):
    from pathlib import Path
    store=LocalSourceStorage(tmp_path)
    original=Path.replace
    def crash(path,target):
        if path.name.endswith('.pending') and '.' not in Path(target).name:
            raise OSError('synthetic process death before data publication')
        return original(path,target)
    with monkeypatch.context() as patch:
        patch.setattr(Path,'replace',crash)
        with pytest.raises(OSError):store.put(19,b'original bytes','image/png','stable-operation')
    refs=store.orphan_candidates(19,float('inf'))
    assert len(refs)==1 and store.orphan_candidates(20,float('inf'))==[]
    ref=store.put(19,b'original bytes','image/png','stable-operation')
    assert ref==refs[0] and store.read(19,ref,100)==b'original bytes'
    store.delete(19,ref,ref.sha256)
    assert store.orphan_candidates(19,float('inf'))==[]
