"""Tests for ``dashboard_save`` -- the call that puts a composed page IN FORCE.

There is no preview gate: the owner crewmate's package is stored and
drawn at once, as an instance version, and one ``dashboard_rollback`` goes back to
any earlier version -- another package version or the template page.

The binding is derived from the vetted caller on the server and never read out of
the body, so a save only ever writes the caller's own package.

Nothing here reimplements the store's own write gate. The save runs
``canonical_package_content`` -- the same function ``ArtifactStore.create`` and
``.update`` run -- and the refusal tests send packages each guard refuses and
assert the save answers 400 with nothing stored.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kiro_crew.artifact_store.dashboard_package import DASHBOARD_KIND, parse_package
from kiro_crew.dashboard.handlers import agent_panel as routes

pytestmark = pytest.mark.asyncio

CREW = "fleet-crew"
SLUG = "fleet-crew"
OTHER_SLUG = "other-crew"


def _package_body() -> dict[str, Any]:
    """The three declarations a save takes. No ``kind``, and no ``bound_to``."""
    return {
        "model": {
            "types": {
                "open_prs": {
                    "type": "number",
                    "source": {"agentic": True},
                    "label": "Open PRs",
                },
                "headline": {"type": "text", "source": {"agentic": True}},
            }
        },
        "view": {
            "blocks": [
                {"id": "lead", "type": "stat", "fields": ["open_prs"], "title": "Open"},
                {"id": "note", "type": "list", "fields": ["headline"]},
            ]
        },
        "theme": {"tokens": {"--accent": "oklch(0.7 0.15 150)"}},
    }


class _Store:
    """A stand-in for ``ArtifactStore`` that keeps the real write gate.

    The gate is the point of the test, so this does NOT fake it: ``create`` and
    ``update`` call ``parse_package`` exactly where the real store calls
    ``canonical_package_content``, and raise the same
    ``ArtifactValidationError``. What is faked is only the filesystem -- the
    records live in a dict -- so a route test needs no artifact home.
    """

    def __init__(self) -> None:
        self.records: dict[str, SimpleNamespace] = {}
        self.updates: list[dict[str, Any]] = []
        #: Slugs whose ``get`` fails. A record that cannot be OPENED is the case
        #: the binding lookup must not answer "absent" for, because it may be the
        #: one bound here -- that is exactly what could not be read.
        self.unreadable: set[str] = set()

    # -- the surface the handler uses -------------------------------------- #

    def list(self, *, kind: str | None = None, **_kw: Any) -> list[SimpleNamespace]:
        return [r for r in self.records.values() if kind is None or r.kind == kind]

    def get(self, slug: str, **_kw: Any) -> SimpleNamespace:
        from kiro_crew.artifact_store.model import ArtifactNotFoundError

        if slug in self.unreadable:
            from kiro_crew.artifact_store.model import ArtifactError

            raise ArtifactError(f"{slug} cannot be opened")
        if slug not in self.records:
            raise ArtifactNotFoundError(slug)
        return self.records[slug]

    def create(self, *, name: str, content: str, kind: str | None = None, **_kw: Any):
        from kiro_crew.artifact_store.dashboard_package import canonical_package_content

        if kind == DASHBOARD_KIND:
            content = canonical_package_content(content)
        slug = f"dashboard-{len(self.records)}"
        rec = SimpleNamespace(
            slug=slug,
            name=name,
            content=content,
            kind=kind or "markdown",
            version=1,
            content_token="0" * 64,
        )
        self.records[slug] = rec
        return rec

    def update(self, slug: str, *, content: str | None = None, **kw: Any):
        from kiro_crew.artifact_store.dashboard_package import (
            canonical_package_content,
            layout_changed,
        )

        rec = self.get(slug)
        self.updates.append({"slug": slug, **kw})
        if content is not None:
            new = canonical_package_content(content)
            if layout_changed(rec.content, new):
                rec.version += 1
            rec.content = new
        return rec


@pytest.fixture
def store(monkeypatch) -> _Store:
    s = _Store()
    monkeypatch.setattr(routes, "_dashboard_artifact_store", lambda: s, raising=False)
    return s


@asynccontextmanager
async def _client(monkeypatch, *, resolved: tuple[str, str, str] | None = (SLUG, CREW, "slot-1")):
    """The panel routes with the caller already vetted, or already refused.

    ``resolved=None`` is the shape every caller the shared gate turns away has --
    a subagent with no dashboard slot, a cookie-only caller, an app identity, a
    restricted session. Those refusals are ``_resolve_dashboard_caller``'s and are
    proven on its own tests; what this file has to prove is that a save ASKS it
    and writes nothing when it says no.
    """

    async def _resolve(_request, _operation):
        if resolved is None:
            return None, web.json_response(
                {"error": "not a dashboard thread", "code": "no_dashboard_slot"}, status=400
            )
        return resolved, None

    monkeypatch.setattr(routes, "_resolve_dashboard_caller", _resolve)
    app = web.Application()
    app["state"] = SimpleNamespace(broadcast_ws=lambda *_a, **_k: None)
    routes.register_agent_panel_routes(app)
    c = TestClient(TestServer(app))
    await c.start_server()
    try:
        yield c
    finally:
        await c.close()


async def _save(client: TestClient, body: dict[str, Any]):
    return await client.post("/api/agent-panel/dashboard/save", json=body)


async def _save_and_apply(client: TestClient, body: dict[str, Any]) -> None:
    """Save a page, then put it live: the package flow is save -> apply."""
    saved = await _save(client, body)
    assert saved.status == 200, await saved.text()
    applied = await _apply(client)
    assert applied.status == 200, await applied.text()


async def _apply(client: TestClient):
    """The person's "yes". Takes no body, like the route."""
    return await client.post("/api/agent-panel/dashboard/apply", json={})


# --------------------------------------------------------------------------- #
# 0. A save puts the page in force, as a version
# --------------------------------------------------------------------------- #


def _instance():
    from kiro_crew.dashboard_templates import instance as instance_store

    return instance_store


async def test_a_save_stores_the_page_and_apply_puts_it_live(monkeypatch, store):
    """Save stores the caller's own package; apply puts it live, with no preview step.

    The package is bound to the CALLER, and the instance version the apply writes
    pins it, so the page read and the write check both draw it.
    """
    async with _client(monkeypatch) as client:
        resp = await _save(client, _package_body())
        assert resp.status == 200, await resp.text()
        payload = await resp.json()
        # Stored, not live yet.
        assert _instance().read(SLUG).instance_version == 0
        applied = await _apply(client)
        assert applied.status == 200, await applied.text()
        reply = await applied.json()

    saved = payload["saved"]
    assert (saved["fields"], saved["blocks"]) == (2, 2)
    [record] = store.records.values()
    assert record.kind == DASHBOARD_KIND
    assert parse_package(record.content)["bound_to"] == f"crewmate:{SLUG}"
    assert saved["package"] == reply["package"] == record.slug
    current = _instance().read(SLUG)
    assert (current.package_slug, current.package_version) == (record.slug, 1)
    assert current.instance_version == reply["instance_version"] == 1


async def test_a_second_save_updates_the_same_package_under_its_read(monkeypatch, store):
    """One package per crewmate: a second save updates it, fenced on the read."""
    second = _package_body()
    second["view"]["blocks"].reverse()
    async with _client(monkeypatch) as client:
        await _save_and_apply(client, _package_body())
        await _save_and_apply(client, second)

    assert len(store.records) == 1
    [update] = store.updates
    assert "snapshot_token" in update
    current = _instance().read(SLUG)
    assert (current.package_version, current.instance_version) == (2, 2)


async def test_one_rollback_goes_back_to_a_package_or_to_the_template(monkeypatch, store):
    """Every apply is a version, and the version restores its own page in force."""
    instance_store = _instance()
    second = _package_body()
    second["view"]["blocks"].reverse()
    async with _client(monkeypatch) as client:
        await _save_and_apply(client, _package_body())  # v1: package v1
        await _save_and_apply(client, second)  # v2: package v2
        applied = await client.post(
            "/api/agent-panel/dashboard/apply", json={"template_id": "project-report"}
        )
        assert applied.status == 200, await applied.text()  # v3: template
        assert instance_store.read(SLUG).package_released is True
        back = await client.post("/api/agent-panel/dashboard/rollback", json={"to_version": 1})
        assert back.status == 200, await back.text()  # v4: package v1 again

    current = instance_store.read(SLUG)
    assert (current.instance_version, current.package_version) == (4, 1)
    assert current.package_released is False


async def test_the_page_read_draws_what_the_record_puts_in_force(monkeypatch, store):
    """The page read follows the record: pinned, locked, then released."""
    from kiro_crew.dashboard.handlers import member_dashboard

    async with _client(monkeypatch) as client:
        await _save_and_apply(client, _package_body())
        [slug] = store.records
        pinned, older = member_dashboard._package_in_force(SLUG)
        assert pinned is not None and (pinned.slug, pinned.version) == (slug, 1)
        assert older is False
        # The lock keeps the pinned page the person locked.
        _instance().set_structure_lock(SLUG, True)
        assert member_dashboard._package_in_force(SLUG)[0] == pinned
        _instance().set_structure_lock(SLUG, False)
        applied = await client.post(
            "/api/agent-panel/dashboard/apply", json={"template_id": "project-report"}
        )
        assert applied.status == 200, await applied.text()
        assert member_dashboard._package_in_force(SLUG) == (None, False)


async def test_the_pinned_page_survives_whatever_happens_to_the_artifact(monkeypatch, store):
    """The page in force is the gateway's snapshot of the saved bytes.

    An artifact deleted and saved again under the same slug, or a pinned artifact
    version the store has pruned, does not change or drop the page a version pins.
    """
    async with _client(monkeypatch) as client:
        await _save_and_apply(client, _package_body())
    [slug] = store.records
    saved = store.records[slug].content
    # Deleted, and a different page saved under the same slug and version.
    other = _package_body()
    other["view"]["blocks"].reverse()
    other.update({"kind": DASHBOARD_KIND, "bound_to": f"crewmate:{SLUG}"})
    store.records.clear()
    store.create(name="again", content=json.dumps(other), kind=DASHBOARD_KIND)
    pinned = _instance().pinned_package(_instance().read(SLUG))
    assert pinned is not None and pinned.content == saved


async def test_only_pinned_pages_are_kept_after_a_save(monkeypatch, store):
    """A save prunes the snapshots no kept version pins, and keeps every pinned one."""
    from kiro_crew.agent_panel import panel_dir

    second = _package_body()
    second["view"]["blocks"].reverse()
    async with _client(monkeypatch) as client:
        await _save_and_apply(client, _package_body())
        await _save_and_apply(client, second)
    pages = sorted(p.name for p in panel_dir().glob(f"{SLUG}.dashboard-page.*.json"))
    # Both versions are kept, so both pages stay reachable by a rollback.
    assert len(pages) == 2, pages
    # A page nobody pins is dropped on the next save.
    stray = panel_dir() / f"{SLUG}.dashboard-page.{'0' * 32}.json"
    stray.write_text("{}", encoding="utf-8")
    async with _client(monkeypatch) as client:
        await _save_and_apply(client, _package_body())
    assert not stray.exists()


async def test_an_older_locked_record_checks_writes_against_its_template(monkeypatch, tmp_path):
    """A record from before pins keeps its template under the lock, for the write too."""
    from kiro_crew import artifacts
    from kiro_crew import dashboard_package as reader
    from kiro_crew.artifacts import ArtifactStore

    instance_store = _instance()
    real = ArtifactStore(tmp_path / "artifacts")
    monkeypatch.setattr(artifacts, "get_default_store", lambda: real)
    instance_store.adopt(SLUG, "project-report")
    # Written before the `package` key existed.
    path = instance_store.instance_dir(SLUG) / "instance.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("package")
    path.write_text(json.dumps(raw), encoding="utf-8")
    # A package bound to the crewmate, with a version a person cut.
    pkg = {**_package_body(), "kind": DASHBOARD_KIND, "bound_to": f"crewmate:{SLUG}"}
    art = real.create(name="page", content=json.dumps(pkg), kind=DASHBOARD_KIND)
    pkg["view"]["blocks"].reverse()
    read = real.get(art.slug, assign_token=True)
    real.update(art.slug, content=json.dumps(pkg), actor="user", snapshot_token=read.content_token)

    assert reader.read_package_model(SLUG).from_package is True
    instance_store.set_structure_lock(SLUG, True)
    locked = reader.read_package_model(SLUG)
    assert locked.from_package is False
    assert locked.model is not None and locked.model.manifest.id == "project-report"


def test_a_package_enum_keeps_its_choices_for_the_write_check():
    """A value outside an `enum` field's declared choices is refused before it lands."""
    from kiro_crew import dashboard_agentic
    from kiro_crew.dashboard.handlers.member_dashboard_fields import package_manifest

    package = {
        "model": {
            "types": {
                "state": {
                    "type": "enum",
                    "choices": ["green", "red"],
                    "source": {"agentic": True},
                }
            }
        }
    }
    model = dashboard_agentic.Instance(
        manifest=package_manifest(package, slug="p", version=1), instance_version=1
    )
    dashboard_agentic.check_write(model, "state", "green", {})
    with pytest.raises(dashboard_agentic.WriteRefused):
        dashboard_agentic.check_write(model, "state", "purple", {})


async def test_a_package_rebound_before_the_fenced_read_is_not_written(monkeypatch, store):
    """The save re-checks the binding on the read whose token fences the update."""
    async with _client(monkeypatch) as client:
        assert (await _save(client, _package_body())).status == 200
        [slug] = store.records
        # Rebound to another crewmate between the lookup and the fenced read.
        theirs = {**_package_body(), "kind": DASHBOARD_KIND, "bound_to": f"crewmate:{OTHER_SLUG}"}
        from kiro_crew.artifact_store.dashboard_package import canonical_package_content

        real_get = store.get

        def _rebound_get(s: str, **kw: Any) -> SimpleNamespace:
            rec = real_get(s, **kw)
            if kw.get("assign_token"):
                rec.content = canonical_package_content(json.dumps(theirs))
            return rec

        monkeypatch.setattr(store, "get", _rebound_get)
        resp = await _save(client, _package_body())
        body = await resp.json()
    assert resp.status == 400, body
    assert "not bound to you" in body["error"]
    assert store.updates == []
    assert slug in store.records


async def test_any_apply_drops_a_preview_left_staged(monkeypatch, store):
    """A committed version replaces a staged look, so the tab never offers a stale Keep."""
    instance_store = _instance()
    async with _client(monkeypatch) as client:
        instance_store.stage_preview(SLUG, template_id="project-report")
        await _save_and_apply(client, _package_body())
        assert instance_store.staged_preview(SLUG) is None
        instance_store.stage_preview(SLUG, template_id="project-report")
        applied = await client.post(
            "/api/agent-panel/dashboard/apply", json={"template_id": "project-report"}
        )
        assert applied.status == 200, await applied.text()
        assert instance_store.staged_preview(SLUG) is None


async def test_a_scan_that_cannot_read_a_kept_version_prunes_nothing(monkeypatch, store):
    """A kept version that will not read may pin a page, so nothing is deleted."""
    from kiro_crew.agent_panel import panel_dir

    instance_store = _instance()
    second = _package_body()
    second["view"]["blocks"].reverse()
    async with _client(monkeypatch) as client:
        await _save_and_apply(client, _package_body())
        stray = panel_dir() / f"{SLUG}.dashboard-page.{'0' * 32}.json"
        stray.write_text("{}", encoding="utf-8")

        def _unreadable(_slug: str, _version: int) -> dict:
            raise instance_store.InstanceError("instance version cannot be read")

        monkeypatch.setattr(instance_store, "_read_version", _unreadable)
        await _save_and_apply(client, second)
    assert stray.exists()
    assert len(list(panel_dir().glob(f"{SLUG}.dashboard-page.*.json"))) == 3


def test_a_concurrent_prune_never_drops_the_page_being_committed(monkeypatch):
    """The snapshot write, the commit and the prune share one lock."""
    instance_store = _instance()
    seen: list[bool] = []
    real_prune = instance_store._prune_page_snapshots

    def _prune(slug: str) -> None:
        # Asked from inside the commit's critical section: the record already pins it.
        record = instance_store._read_record(slug) or {}
        seen.append(bool(instance_store._package_field(record)[2]))
        real_prune(slug)

    monkeypatch.setattr(instance_store, "_prune_page_snapshots", _prune)
    pkg = {**_package_body(), "kind": DASHBOARD_KIND, "bound_to": f"crewmate:{SLUG}"}
    from kiro_crew.artifact_store.dashboard_package import canonical_package_content

    record = instance_store.apply_package(
        SLUG,
        package_slug="p",
        package_version=1,
        content=canonical_package_content(json.dumps(pkg)),
    )
    assert seen == [True]
    assert instance_store.pinned_package(record) is not None


async def test_a_template_applies_without_a_preview(monkeypatch, store):
    """A preview is an optional look: ``template_id`` applies directly."""
    async with _client(monkeypatch) as client:
        resp = await client.post(
            "/api/agent-panel/dashboard/apply", json={"template_id": "project-report"}
        )
        assert resp.status == 200, await resp.text()
        assert (await resp.json())["template_id"] == "project-report"
        unknown = await client.post(
            "/api/agent-panel/dashboard/apply", json={"template_id": "no-such-page"}
        )
        assert unknown.status == 400, await unknown.text()
    assert _instance().read(SLUG).instance_version == 1


async def test_a_root_session_save_is_refused_and_writes_nothing(monkeypatch, store):
    """The hosted-page reader resolves a crewmate's binding only, so nothing would draw."""
    session_slug = _instance().session_instance_key("chat-7-17910")
    async with _client(monkeypatch, resolved=(session_slug, "", "chat-7-17910")) as client:
        resp = await _save(client, _package_body())
        body = await resp.json()
    assert resp.status == 400, body
    assert body["code"] == "session_package_unsupported"
    assert store.records == {}


async def test_a_locked_layout_refuses_the_apply_and_keeps_the_page(monkeypatch, store):
    """The lock refuses the apply; the save is only a write to the crewmate's artifact."""
    instance_store = _instance()
    instance_store.set_structure_lock(SLUG, True)
    async with _client(monkeypatch) as client:
        assert (await _save(client, _package_body())).status == 200
        resp = await _apply(client)
        body = await resp.json()
    assert resp.status == 423, body
    assert body["code"] == "dashboard_locked"
    assert instance_store.read(SLUG).package_slug == ""


async def test_dashboard_write_is_checked_against_the_package_in_force(monkeypatch, tmp_path):
    """The write guard is lifted: a bound package's own Model checks the write."""
    from kiro_crew import artifacts, dashboard_agentic
    from kiro_crew import dashboard_package as reader
    from kiro_crew.artifacts import ArtifactStore

    real = ArtifactStore(tmp_path / "artifacts")
    monkeypatch.setattr(routes, "_dashboard_artifact_store", lambda: real)
    monkeypatch.setattr(artifacts, "get_default_store", lambda: real)
    async with _client(monkeypatch) as client:
        await _save_and_apply(client, _package_body())

    model = reader.read_package_model(SLUG)
    assert model.from_package is True and model.model is not None
    assert set(model.model.manifest.fields) == {"open_prs", "headline"}
    dashboard_agentic.check_write(model.model, "open_prs", 3, {})
    with pytest.raises(dashboard_agentic.WriteRefused):
        dashboard_agentic.check_write(model.model, "open_prs", "three", {})


async def test_an_apply_with_nothing_staged_writes_nothing(monkeypatch, store):
    """No template named and nothing staged: the apply has nothing to put in force."""
    async with _client(monkeypatch) as client:
        resp = await _apply(client)
        assert resp.status == 400, await resp.text()

    assert store.records == {}
    assert store.updates == []


# --------------------------------------------------------------------------- #
# 1. An applied page round-trips
# --------------------------------------------------------------------------- #


async def test_the_binding_comes_from_the_caller_and_the_body_cannot_carry_one(monkeypatch, store):
    """``bound_to`` in the body is refused, not merged and not ignored.

    Ignoring it would be safe and still wrong: an agent that sent one would be
    told its save succeeded while the page went somewhere else, and would send it
    again every cycle. The refusal is the only answer it can act on.
    """
    body = _package_body()
    body["bound_to"] = f"crewmate:{OTHER_SLUG}"
    async with _client(monkeypatch) as client:
        resp = await _save(client, body)
        assert resp.status == 400
        payload = await resp.json()
    assert "bound_to" in payload["error"]
    assert store.records == {}


# --------------------------------------------------------------------------- #
# 2. An unchanged layout cuts no version
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# 3. An invalid package is refused with a 400-class error
# --------------------------------------------------------------------------- #


async def test_an_unknown_block_type_is_refused(monkeypatch, store):
    body = _package_body()
    body["view"]["blocks"][0]["type"] = "hologram"
    async with _client(monkeypatch) as client:
        resp = await _save(client, body)
        assert resp.status == 400
        payload = await resp.json()
    assert "hologram" in payload["error"]
    assert store.records == {}


async def test_an_oversized_package_is_refused(monkeypatch, store):
    """Over the package cap, refused as a package and not as a 500."""
    body = _package_body()
    body["model"]["types"]["open_prs"]["description"] = "x" * 400_000
    async with _client(monkeypatch) as client:
        resp = await _save(client, body)
        assert resp.status == 400
        payload = await resp.json()
    assert payload["code"] == "invalid_package"
    assert store.records == {}


def _lone_surrogate_body() -> bytes:
    """A package whose block title is an unpaired ``\\ud800``, as raw JSON bytes.

    Raw rather than through ``json=``: the escape has to reach the handler AS
    JSON, because it is six ASCII characters until the decoder turns it into the
    single code point no UTF-8 encoder accepts.
    """
    body = _package_body()
    body["view"]["blocks"][0]["title"] = "LONE"
    return json.dumps(body).replace("LONE", "\\ud800").encode("utf-8")


async def test_a_lone_surrogate_is_refused(monkeypatch, store):
    async with _client(monkeypatch) as client:
        resp = await client.post(
            "/api/agent-panel/dashboard/save",
            data=_lone_surrogate_body(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 400
    assert store.records == {}


async def test_a_lone_surrogate_never_reaches_the_real_store_either(monkeypatch, tmp_path):
    """The same body with a REAL store behind the route, which it must never reach.

    The refusal is the save's now, because the save runs the dashboard gate itself
    and nothing is handed to a store until an apply. So the thing worth pinning
    with a real store wired up is that the store stays EMPTY: no record, and no
    apply to make one, because the package was refused while the agent was still
    composing. The store's own refusal, for every other caller, is pinned in
    ``test_artifact_update_refuses_a_surrogate_the_same_way``.
    """
    from kiro_crew.artifacts import ArtifactStore

    real = ArtifactStore(tmp_path / "artifacts")
    monkeypatch.setattr(routes, "_dashboard_artifact_store", lambda: real, raising=False)
    async with _client(monkeypatch) as client:
        resp = await client.post(
            "/api/agent-panel/dashboard/save",
            data=_lone_surrogate_body(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 400, await resp.text()
        assert (await resp.json())["code"] == "invalid_package"
        # And nothing was staged for an apply to pick up.
        assert (await _apply(client)).status == 400
    assert real.list(kind=DASHBOARD_KIND) == []


async def test_artifact_update_refuses_a_surrogate_the_same_way(tmp_path):
    """The store's OWN refusal, on ``update`` as well as ``create``.

    The clean refusal belongs to the store, so it holds for every caller --
    including a plain ``artifact_update`` the save route never touches.
    ``_validate_content`` is the first call in both methods and it measures
    content by encoding it, and content holding an unpaired surrogate cannot be
    encoded at all: answering that with ``UnicodeEncodeError`` would turn an
    artifact a caller can fix into a 500.
    """
    from kiro_crew.artifact_store.model import ArtifactValidationError
    from kiro_crew.artifacts import ArtifactStore

    store = ArtifactStore(tmp_path / "artifacts")
    pkg = _package_body()
    pkg["kind"] = DASHBOARD_KIND
    pkg["bound_to"] = f"crewmate:{SLUG}"
    created = store.create(
        name="page", content=json.dumps(pkg), kind=DASHBOARD_KIND, source="dashboard"
    )

    # A REAL lone code point in the content text, which is what reaches the store:
    # the escape ``\\ud800`` is six ASCII characters and encodes fine, so a package
    # built by replacing text with it would exercise the decoder's refusal instead
    # and pass whatever ``_validate_content`` does. ``ensure_ascii=False`` is what
    # keeps the code point itself in the output, and is what the save route uses.
    surrogate = dict(pkg)
    surrogate["view"] = {"blocks": [dict(pkg["view"]["blocks"][0], title="\ud800")]}
    bad = json.dumps(surrogate, ensure_ascii=False)
    assert "\ud800" in bad
    # UNDER THE RECORD'S OWN TOKEN, the way an owner write reaches a dashboard
    # record: `update` refuses a content write on `kind="dashboard"` that does not
    # name the read it was authorized against, and an absent token is its own
    # refusal (`ArtifactConflictError`). Passing it keeps this test pointed at the
    # surrogate refusal rather than at the owner gate, which has its own tests.
    token = store.get(created.slug, assign_token=True).content_token
    with pytest.raises(ArtifactValidationError) as update_refusal:
        store.update(created.slug, content=bad, snapshot_token=token)
    assert "surrogate" in str(update_refusal.value)

    with pytest.raises(ArtifactValidationError) as create_refusal:
        store.create(name="other", content=bad, kind=DASHBOARD_KIND, source="dashboard")
    assert "surrogate" in str(create_refusal.value)

    # The live package is untouched by the refused update, and the control: the
    # same bytes without the surrogate are still accepted, so the refusal above
    # is the surrogate's and not the whole package being rejected.
    assert parse_package(store.get(created.slug).content)["bound_to"] == f"crewmate:{SLUG}"
    live = store.get(created.slug, assign_token=True)
    assert store.update(created.slug, content=json.dumps(pkg), snapshot_token=live.content_token)


async def test_a_credential_shaped_field_name_is_refused_at_save(monkeypatch, store):
    """A name a redactor would mask is refused HERE, and nothing is written.

    The field-name grammar both sides enforce, ``[a-z][a-z0-9_]{0,63}``, admits a
    credential shape: ``ghp_`` plus 36 lowercase characters satisfies it. The page
    refuses a read whose declared names a redactor would rewrite, and it refuses
    the WHOLE read, because a field name is the key joining the read's fields, the
    frame's blocks and the page's cells. So a save that stored such a name would
    succeed and hand the crewmate a page that can never render -- while this
    route's reply says the tab draws it. That reply would be false, so the save
    refuses instead.

    The premise is asserted on the REDACTOR rather than on a hand-written pattern,
    because the redactor is what both sides consult. A test that invented its own
    pattern would pin a rule nothing enforces.

    The refusal names a POSITION, not the name. The page's own counter withholds
    the offending name on purpose, and the MCP boundary redacts this sentence
    anyway, so a name would reach the agent masked while travelling raw through the
    gateway's logs. A position resolves against the package in the agent's own hand.
    """
    from kiro_crew.platform import redact_via_context

    token_name = "ghp_" + "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8"
    # The two halves of the premise, each asserted rather than assumed.
    assert redact_via_context(token_name) != token_name, "not credential-shaped here"
    assert redact_via_context("open_prs") == "open_prs", "control name is not redacted"

    body = _package_body()
    body["model"]["types"][token_name] = {"type": "number", "source": {"agentic": True}}
    body["view"]["blocks"][0]["fields"] = [token_name]

    async with _client(monkeypatch) as client:
        resp = await _save(client, body)
        assert resp.status == 400, await resp.text()
        payload = await resp.json()

    assert payload["code"] == "unsafe_declared_name"
    assert "looks like a credential" in payload["error"]
    # A position the agent can resolve, and NOT the name itself.
    assert "declared field #" in payload["error"]
    assert token_name not in payload["error"], "the refusal echoed the name back"
    # NOTHING WRITTEN. A refusal that minted the artifact first would be a hole the
    # status code hides, which is the assertion every other refusal here makes.
    assert store.records == {}
    assert store.updates == []


async def test_an_ordinary_field_name_still_saves(monkeypatch, store):
    """The control for the refusal above, and why it is not a blanket block.

    Without it, a route that refused EVERY package would pass that test. The body
    is this file's ordinary one, whose names no redactor touches.
    """
    async with _client(monkeypatch) as client:
        resp = await _save(client, _package_body())
        assert resp.status == 200, await resp.text()

    assert len(store.records) == 1


async def test_a_credential_shaped_block_id_is_refused_too(monkeypatch, store):
    """The other name set the page counts: a view block's id.

    Both reach a page as keys -- a field name through the read and the manifest, a
    block id through the frame's ``blocks`` mapping -- so a check catching only one
    would leave the same hole one key away.
    """
    from kiro_crew.platform import redact_via_context

    token_id = "ghp_" + "f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1"
    assert redact_via_context(token_id) != token_id, "not credential-shaped here"

    body = _package_body()
    body["view"]["blocks"][0]["id"] = token_id

    async with _client(monkeypatch) as client:
        resp = await _save(client, body)
        assert resp.status == 400, await resp.text()
        payload = await resp.json()

    assert payload["code"] == "unsafe_declared_name"
    assert "view block #" in payload["error"]
    assert token_id not in payload["error"]
    assert store.records == {}


async def test_a_credential_shaped_theme_token_is_refused_too(monkeypatch, store):
    """The THIRD name set a package carries, and the one whose grammar hides it.

    A field name and a block id match ``[a-z][a-z0-9_]{0,63}``. A theme token is a
    CSS custom property, ``--[a-z][a-z0-9-]{0,63}``, so a lowercase HYPHENATED
    look-alike satisfies it where an underscored one never could -- which is how a
    check written for the first two name sets leaves this one open.

    Refused rather than redacted on the way out, for the reason the other two are:
    a token key is a JOIN KEY. The page's own style references the variable by
    name, so a redactor rewriting it hands back a stylesheet pointing at a property
    that does not exist. There is no masked form that still works.
    """
    from kiro_crew.platform import redact_via_context

    token_key = "--xoxb-" + "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8"
    # The two halves of the premise, each asserted rather than assumed.
    assert redact_via_context(token_key) != token_key, "not credential-shaped here"
    assert redact_via_context("--accent") == "--accent", "control token is not redacted"

    body = _package_body()
    body["theme"]["tokens"][token_key] = "oklch(0.5 0.1 200)"

    async with _client(monkeypatch) as client:
        resp = await _save(client, body)
        assert resp.status == 400, await resp.text()
        payload = await resp.json()

    assert payload["code"] == "unsafe_declared_name"
    # A position the agent can resolve, and NOT the name itself.
    assert "theme token #" in payload["error"]
    assert token_key not in payload["error"], "the refusal echoed the name back"
    # NOTHING STAGED and nothing stored, so no apply can reach it either.
    assert store.records == {}
    async with _client(monkeypatch) as client:
        assert (await _apply(client)).status == 400


async def test_an_ordinary_theme_token_still_saves(monkeypatch, store):
    """The control: without it, a route refusing EVERY theme would pass above."""
    body = _package_body()
    body["theme"]["tokens"]["--surface"] = "oklch(0.98 0.01 250)"
    async with _client(monkeypatch) as client:
        resp = await _save(client, body)
        assert resp.status == 200, await resp.text()

    assert len(store.records) == 1


async def test_the_real_store_round_trips_a_valid_package(monkeypatch, tmp_path):
    """The positive control for the surrogate test, through the real store.

    Without it, a route that refused EVERY package would pass the surrogate test,
    so this proves the 400 there came from the surrogate and not from the save path
    being broken against a real store.
    """
    from kiro_crew.artifacts import ArtifactStore

    real = ArtifactStore(tmp_path / "artifacts")
    monkeypatch.setattr(routes, "_dashboard_artifact_store", lambda: real)
    async with _client(monkeypatch) as client:
        resp = await _save(client, _package_body())
        assert resp.status == 200, await resp.text()
        assert "saved" in await resp.json()

    [stored] = real.list(kind=DASHBOARD_KIND)
    assert parse_package(real.get(stored.slug).content)["bound_to"] == f"crewmate:{SLUG}"


# --------------------------------------------------------------------------- #
# 4. A caller the shared gate turns away saves nothing
# --------------------------------------------------------------------------- #


async def test_a_caller_with_no_dashboard_slot_is_refused(monkeypatch, store):
    """The subagent case. A subagent holds no slot, so the shared gate refuses it.

    Asserting the STORE stayed empty is the part that matters: a save that
    answered 400 after minting the artifact would be a hole the status code hides.
    """
    async with _client(monkeypatch, resolved=None) as client:
        resp = await _save(client, _package_body())
        assert resp.status == 400
        assert (await resp.json())["code"] == "no_dashboard_slot"
    assert store.records == {}


# --------------------------------------------------------------------------- #
# 5. A caller cannot save into another crewmate's panel
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# The MCP tool half
# --------------------------------------------------------------------------- #


def test_the_tool_is_on_the_panel_server_and_takes_no_target():
    from kiro_crew import mcp_panel

    tool = next(t for t in mcp_panel._tool_definitions() if t["name"] == "dashboard_save")
    props = tool["inputSchema"]["properties"]
    assert set(props) == {"model", "view", "theme"}
    assert sorted(tool["inputSchema"]["required"]) == ["model", "theme", "view"]
    assert not (set(props) & {"bound_to", "session", "session_key", "slot", "target", "kind"})


def test_the_tool_refuses_a_caller_whose_identity_is_not_strict(monkeypatch):
    """A subagent, answered by the tool before any request leaves the process."""
    from kiro_crew import mcp_panel

    monkeypatch.setattr(mcp_panel, "_strict_session_key", lambda: ("", "Error: no identity"))

    def _boom(*_a: Any, **_k: Any):  # pragma: no cover - must not run
        raise AssertionError("a caller with no strict identity reached the gateway")

    monkeypatch.setattr(mcp_panel, "_post", _boom)
    out = mcp_panel._call_tool_inner("dashboard_save", _package_body())
    assert out == "Error: no identity"


def test_the_tool_sends_only_the_three_declarations(monkeypatch):
    from kiro_crew import mcp_panel

    sent: dict[str, Any] = {}
    monkeypatch.setattr(mcp_panel, "_strict_session_key", lambda: ("sk", ""))

    def _post(path: str, payload: dict[str, Any], **_kw: Any) -> dict[str, Any]:
        sent["path"] = path
        sent["payload"] = payload
        return {
            "saved": {
                "package": "p",
                "package_version": 1,
                "instance_version": 3,
                "fields": 2,
                "blocks": 2,
            },
            "dashboard": "[DASHBOARD]\nYour Dashboard tab shows the page you saved.",
        }

    monkeypatch.setattr(mcp_panel, "_post", _post)
    body = _package_body()
    body["bound_to"] = "crewmate:someone-else"
    out = mcp_panel._call_tool_inner("dashboard_save", body)
    assert sent["path"] == "/api/agent-panel/dashboard/save"
    assert set(sent["payload"]) == {"model", "view", "theme"}
    # SAVED, not live: the reply names the apply that puts it live.
    assert "SAVED" in out and "not live yet" in out
    assert "dashboard_apply" in out


def test_the_tool_hands_back_a_refusal_whole(monkeypatch):
    """The refusal names the key and the catalog, which is what the agent fixes from."""
    from kiro_crew import mcp_panel

    monkeypatch.setattr(mcp_panel, "_strict_session_key", lambda: ("sk", ""))
    monkeypatch.setattr(
        mcp_panel,
        "_post",
        lambda *_a, **_k: {
            "error": "dashboard package: view.blocks[0].type: unknown block type 'hologram'",
            "code": "invalid_package",
        },
    )
    out = mcp_panel._call_tool_inner("dashboard_save", _package_body())
    assert "hologram" in out
    assert "view.blocks[0].type" in out


# --------------------------------------------------------------------------- #
# 6. The grant equals the registration
# --------------------------------------------------------------------------- #


def test_the_member_grant_names_the_save_tool():
    from kiro_crew.agent import _MEMBER_PANEL_GRANTS

    assert "@kirocrew-panel/dashboard_save" in _MEMBER_PANEL_GRANTS


def test_the_tool_has_a_registered_argument_schema():
    from kiro_crew.validation import MCP_PANEL_SCHEMAS

    assert "dashboard_save" in MCP_PANEL_SCHEMAS
