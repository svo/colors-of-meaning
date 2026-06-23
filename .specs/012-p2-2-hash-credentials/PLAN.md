# Plan: hash-credentials

## Implementation Strategy

The defect is twofold: `BasicAuthenticator` stores plaintext passwords and
verifies them with a non-constant-time `==` (`infrastructure/security/basic_authentication.py`
lines 17, 25), and a working credential is committed to
`resources/application.properties` (lines 1-2) and shipped via the `MANIFEST.in`
graft. The fix replaces store-and-compare with verify-against-an-Argon2-hash
using `argon2-cffi`'s constant-time `PasswordHasher.verify`, removes the
plaintext lines from the properties file, deletes the plaintext `password`
default from `ApplicationSettings`, and seeds the authenticator from an
environment-sourced password **hash** (`APP_ADMIN_PASSWORD_HASH`). Tests compute
a known hash at test time so nothing secret is committed; a documented local-dev
step lets developers export a throwaway hash.

Work proceeds test-first (TDD) in dependency order — domain (confirm the port is
hash-agnostic), application (no change, guard with archon), shared config,
infrastructure verifier, interface wiring/dev helper, then cross-cutting tests —
with each test written before its implementation. No auth test performs network
or file I/O for secrets: hashes are generated in fixtures via
`PasswordHasher().hash(...)`. The new dependency `argon2-cffi` is added to
`setup.cfg` `install_requires`; `pip-audit` under `tox` must stay green. The
existing `SecurityDependency` HTTP behaviour (200 valid / 401 + `WWW-Authenticate:
Basic`) is preserved, so its `TestClient` tests are updated only to register a
hash instead of a plaintext password.

## Layer Changes

### Domain Layer (`src/colors_of_meaning/domain/`)

- `authentication/authenticator.py`: the `Authenticator` ABC keeps
  `verify_credentials(self, username: str, password: str) -> bool` (line 6)
  unchanged — the port already takes a cleartext attempt and lets the
  implementation decide how to check it, so it is hash-agnostic. No hashing
  import is added. (Optionally promote `register_user` onto the ABC with a
  hash-typed parameter — recorded as an Open Question; default is no port change.)

### Application Layer (`src/colors_of_meaning/application/`)

- No changes. No use case touches the authenticator; basic auth is wired in
  interface/infrastructure only. An archon rule guards that application still
  imports domain only.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

- `security/basic_authentication.py`:
  - Introduce a module-level `PasswordHasher` (from `argon2.PasswordHasher`)
    used for verification.
  - Retarget the credential store (`self.user_credentials`, line 14) to hold a
    username-to-hash map (renamed to convey "hash", e.g.
    `self.user_password_hashes`); `register_user` (line 17) stores a hash string.
  - Rewrite `verify_credentials` (lines 19-25): look up the stored hash; if
    `None` return `False`; otherwise return the result of verifying the supplied
    cleartext against the stored hash, catching the library's
    `VerifyMismatchError` (and malformed-hash error) to return `False`. No `==`
    on secret material remains.
  - `get_basic_authenticator` (lines 55-64): read the username via
    `setting_provider.get("admin")` and the hash via
    `setting_provider.get("admin_password_hash")`; register the hash. Remove the
    plaintext `password` read.
  - `SecurityDependency` (lines 28-52) and `get_security_dependency` (lines
    67-68) are unchanged.

### Interface Layer (`src/colors_of_meaning/interface/`)

- `api/main.py`: no structural change; `get_basic_authenticator()` (lines 43-46)
  now returns a hash-seeded authenticator and is registered in the Lagom
  `Container()` exactly as today.
- Optional `cli/hash_password.py`: a small command that prompts for a password
  and prints its Argon2 hash, importing the infrastructure hasher, so developers
  never type a secret into a tracked file (Open Question whether to ship it).

### Shared Layer (`src/colors_of_meaning/shared/`)

- `configuration.py`: delete `password: str = "password"` (line 24); add
  `admin_password_hash: str = ""` populated from `APP_ADMIN_PASSWORD_HASH` via
  the existing `env_prefix="APP_"` (lines 28-32) and overlayable through
  `_apply_property` (lines 53-55). Keep the `admin` username field (line 23).
  Define fail-closed behaviour when the hash is empty/unset (mirroring the
  host-not-set guard at lines 76-77) — exact behaviour per Open Question.
- `resources/application.properties`: remove `admin=admin` and
  `password=password` (lines 1-2); keep `reload=false` and `host=0.0.0.0`
  (lines 3-4).
- `resources/__init__.py` `get_resource_path` (lines 4-12): unchanged.
- `.gitignore`: add an `.env` ignore entry so a developer-exported hash file is
  never accidentally committed (currently only `.venv*/` is ignored).

## Dependency Injection

The authenticator is built by `get_basic_authenticator` and registered in the
API Lagom `Container()` (`interface/api/main.py` lines 43-46) as today; only the
setting it reads changes (a hash instead of a plaintext password). The
`PasswordHasher` is constructed once at module scope (or via a thin infrastructure
provider) and used only inside `basic_authentication.py`; callers never
instantiate it. `argon2-cffi` is declared in `setup.cfg` `install_requires` (no
duplicate pin in `pyproject.toml`, which holds no runtime-deps table); `pip-audit`
under `tox` checks the resolved environment.

## Task List

1. [ ] domain: add a port-contract test that a stub `Authenticator` implementing
   `verify_credentials(username, password) -> bool` is instantiable and callable,
   confirming the port stays hash-agnostic and unchanged.
2. [ ] application: add (or confirm) a `pytest-archon` rule that
   `application.*` imports `domain` only and never imports `infrastructure`,
   `argon2`, or `interface`.
3. [ ] shared: add a failing test that `ApplicationSettings` has no
   `password` attribute (plaintext default removed).
4. [ ] shared: add a failing test that `ApplicationSettings.admin_password_hash`
   is read from `APP_ADMIN_PASSWORD_HASH` when that env var is set (patched env).
5. [ ] shared: add a failing test that `admin_password_hash` is overlaid from a
   properties dict via the existing `_apply_property` path (patched
   `load_properties_file`/`get_resource_path`).
6. [ ] shared: add a failing test for the chosen fail-closed behaviour when
   `admin_password_hash` is empty and no env var is set (raise / refuse), modelled
   on the host-not-set guard test.
7. [ ] shared: implement the `configuration.py` change — delete
   `password: str = "password"`, add `admin_password_hash: str = ""`, wire env +
   properties overlay and the fail-closed guard — passing tasks 3-6.
8. [ ] shared: remove `admin=admin` and `password=password` from
   `resources/application.properties`, keeping `reload`/`host`.
9. [ ] shared: update the existing config tests that reference `password`/the old
   defaults so they assert the new hash field and the credential-free properties
   file; add `.env` to `.gitignore`.
10. [ ] infrastructure: add a failing test that `verify_credentials` returns
    `True` when the supplied password matches a stored Argon2 hash (fixture hash
    via `PasswordHasher().hash("known-test-password")`).
11. [ ] infrastructure: add a failing test that `verify_credentials` returns
    `False` when the password does not match the stored hash.
12. [ ] infrastructure: add a failing test that `verify_credentials` returns
    `False` for an unknown username (no verification performed).
13. [ ] infrastructure: add a failing test that the stored credential value is an
    Argon2 hash (starts with `$argon2`) and is not equal to the plaintext
    password (no plaintext retained).
14. [ ] infrastructure: add a failing test that a malformed/non-Argon2 stored
    value causes `verify_credentials` to return `False` rather than raise (covers
    the verification-exception branch).
15. [ ] infrastructure: rewrite `register_user`/`verify_credentials` and the
    credential store in `basic_authentication.py` to store and constant-time
    `PasswordHasher.verify` against a hash, passing tasks 10-14; add `argon2-cffi`
    to `setup.cfg` `install_requires`.
16. [ ] infrastructure: add a failing test that `get_basic_authenticator` reads
    the username via `get("admin")` and the hash via `get("admin_password_hash")`
    and registers the hash (mock setting provider; assert calls and that no
    plaintext password is read).
17. [ ] infrastructure: implement the `get_basic_authenticator` change to seed
    from the hash, passing task 16.
18. [ ] interface: update the `SecurityDependency` `TestClient` tests to register
    a known hash (fixture) and assert 200 with the matching password, 401 with a
    wrong password, and the `WWW-Authenticate: Basic` header — preserving the
    existing contract against the hash-backed authenticator.
19. [ ] interface: (optional, per Open Question) add `cli/hash_password.py` that
    prints an Argon2 hash for a prompted password, with a test asserting the
    printed value verifies for that password; document the local-dev export step.
20. [ ] tests: add a `pytest-archon` rule that the `argon2` import is confined to
    `infrastructure.security` (and the optional dev CLI), and never appears in
    `domain` or `application`.
21. [ ] tests: add an observability test that a verification emits a structured
    log with `correlation-id` and outcome but never logs the password or the hash
    (assert the secret is absent from the emitted record).
22. [ ] all: run `tox` and resolve every gate (flake8, black, bandit, semgrep,
    pip-audit, radon, xenon, mypy) to green at 100% coverage, confirming bandit
    reports no hardcoded-password (`B105/B106`) or weak-hash findings.

## Testing Strategy

- **Framework split:** base-entity/config assertions use `assertpy`'s
  `assert_that` (consistent with the existing `test_basic_authentication.py` and
  `test_configuration.py`); security/verification behaviour that is effectively
  domain-numerical (verify true/false, exception branch) may use plain `assert`
  and `pytest.raises`. Either style keeps one logical assertion per test.
- **One logical assertion per test:** each property — match-accepts,
  mismatch-rejects, unknown-user-rejects, value-is-a-hash-not-plaintext,
  malformed-hash-rejects, env-hash-read, properties-overlay, fail-closed,
  factory-reads-hash, 200/401/header contract — is its own test.
- **Naming:** every test follows `test_should_<behaviour>_when_<condition>`, e.g.
  `test_should_accept_password_when_it_matches_stored_hash`,
  `test_should_reject_password_when_it_does_not_match_hash`,
  `test_should_store_hash_when_user_is_registered`,
  `test_should_read_hash_from_environment_when_app_admin_password_hash_is_set`,
  `test_should_refuse_authentication_when_password_hash_is_unset`.
- **No committed secret / no network:** hashes are generated in fixtures via
  `PasswordHasher().hash("known-test-password")`; env is provided with
  `patch.dict(os.environ, ...)` exactly as in the existing config tests; no real
  or committed hash is used and no I/O for secrets occurs. Consider reduced
  Argon2 cost parameters in tests to keep the suite fast.
- **Architecture:** `pytest-archon` `archrule`s confirm the `argon2` import stays
  within `infrastructure.security` (plus the optional dev CLI) and that
  `application`/`domain` import neither `argon2` nor `infrastructure`.
- **Contract:** the `SecurityDependency` `TestClient` tests act as the
  producer/CDCT check that the 200-vs-401 + `WWW-Authenticate: Basic` contract
  holds with hash-backed verification.
- **Coverage and gates:** verified only via `tox` (all 8 gates), never `pytest`
  alone; 100% coverage including the unknown-user, mismatch, malformed-hash, and
  fail-closed branches. Bandit must show no `B105/B106` and no weak-hash finding.

## Observability Plan

Emit one structured log per verification (outcome + username + `correlation-id`)
and one startup line stating the authenticator was seeded from an
environment-sourced hash, using the existing `infrastructure/observability/`
logger conventions. The password and the hash are never logged; a dedicated test
asserts the secret is absent from the emitted record. The fail-closed path logs a
warning that authentication is unavailable rather than allowing access.
Optionally add an auth success/failure counter metric. No new secret is
introduced into logs or metrics.

## Risks and Mitigations

- **Risk: existing auth and config tests break.** They register plaintext
  passwords and assert the old `password` default. *Mitigation:* update them to
  register a fixture-generated hash and to assert the new `admin_password_hash`
  field and credential-free properties file (tasks 9, 18); the HTTP contract
  assertions stay.
- **Risk: a usable default credential survives.** *Mitigation:* delete the
  `password` default and the properties lines, and add a fail-closed test so an
  unset hash refuses login rather than falling back to a default (tasks 6-8).
- **Risk: bandit flags hardcoded passwords or weak hashing.** *Mitigation:* no
  password literals in source (test hashes are generated, not literal); use
  `argon2-cffi` (no `hashlib`); task 22 confirms bandit is clean.
- **Risk: new dependency carries a vulnerability or breaks the build.**
  *Mitigation:* add `argon2-cffi` to `setup.cfg` and run `pip-audit` under `tox`
  (task 22); pin/constrain if `pip-audit` reports an advisory.
- **Risk: slow tests from Argon2 cost.** *Mitigation:* generate hashes once per
  session fixture and use reduced cost parameters in the test configuration.
- **Risk: secret leaks via logs or a committed `.env`.** *Mitigation:* never log
  password/hash (task 21) and add `.env` to `.gitignore` (task 9).
- **Risk: plaintext remains in git history after de-committing.** *Mitigation:*
  remove it going forward in this step and record history-scrub / rotation as an
  Open Question tracked in the security backlog, not silently assumed done.
