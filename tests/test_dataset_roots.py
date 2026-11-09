"""Checks that every ``--*-root`` flag reaches the configuration honestly.

Only temporary directories are used: no dataset is created, no manifest is written
and no download is attempted. The point of these tests is that a supplied root is
used verbatim, that a missing or malformed root is reported instead of being
replaced by invented data, and that a missing credential is reported as missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ConfigError, load_config  # noqa: E402
from src.dataset_downloaders import aptos as aptos_adapter  # noqa: E402
from src.dataset_downloaders import idrid as idrid_adapter  # noqa: E402
from src.dataset_downloaders.manager import dataset_inventory  # noqa: E402
from src.pipeline import ROOT_OVERRIDE_FLAGS, apply_root_overrides, build_parser  # noqa: E402

ROOT_ENV_VARS = [env_var for _attribute, env_var, _label in ROOT_OVERRIDE_FLAGS]

READERS = {
    "fundus_root": lambda cfg: cfg.fundus_root(),
    "mmrdr_root": lambda cfg: cfg.mmrdr_root(),
    "idrid_root": lambda cfg: cfg.idrid_root(),
    "aptos_root": lambda cfg: cfg.aptos_root(),
    "ddr_root": lambda cfg: cfg.ddr_root(),
}


@pytest.fixture(autouse=True)
def _clean_root_environment(monkeypatch):
    for name in ROOT_ENV_VARS + ["DR_IDRID_URLS", "DR_IDRID_COOKIE"]:
        monkeypatch.delenv(name, raising=False)
    yield


def _flag(attribute: str) -> str:
    return "--" + attribute.replace("_", "-")


def test_all_five_root_flags_are_registered():
    assert [attribute for attribute, _env, _label in ROOT_OVERRIDE_FLAGS] == [
        "fundus_root",
        "mmrdr_root",
        "idrid_root",
        "aptos_root",
        "ddr_root",
    ]


@pytest.mark.parametrize("attribute", [row[0] for row in ROOT_OVERRIDE_FLAGS])
def test_parser_accepts_each_root_flag(tmp_path, attribute):
    target = tmp_path / "some corpus"
    args = build_parser().parse_args([_flag(attribute), str(target)])
    assert getattr(args, attribute) == str(target)


@pytest.mark.parametrize("attribute,env_var", [(row[0], row[1]) for row in ROOT_OVERRIDE_FLAGS])
def test_root_flag_reaches_the_config_reader(tmp_path, attribute, env_var):
    """A path containing spaces must survive the flag -> env var -> config hop."""
    target = tmp_path / "My Data" / "Fundus Corpus 2019"
    target.mkdir(parents=True)

    args = build_parser().parse_args([_flag(attribute), str(target)])
    applied = apply_root_overrides(args)

    assert applied[env_var] == str(target.resolve())
    cfg = load_config()
    assert Path(READERS[attribute](cfg)) == target.resolve()


def test_unset_fundus_root_stays_unset():
    cfg = load_config()
    # paths.legacy_fundus_root is empty in the shipped configuration.
    assert cfg.fundus_root() is None


def test_empty_root_argument_is_ignored(tmp_path):
    args = build_parser().parse_args(["--fundus-root", ""])
    assert apply_root_overrides(args) == {}
    for name in ROOT_ENV_VARS:
        assert name not in __import__("os").environ


def test_missing_directory_is_accepted_and_reported_not_faked(tmp_path):
    missing = tmp_path / "not_downloaded_yet"
    args = build_parser().parse_args(["--aptos-root", str(missing)])
    apply_root_overrides(args)

    cfg = load_config()
    assert cfg.aptos_root() == missing.resolve()
    assert not cfg.aptos_root().is_dir()

    inventory = dataset_inventory(cfg)
    assert inventory["aptos"]["root"] == str(missing.resolve())
    assert inventory["aptos"]["images"] == 0
    # No manifest or placeholder dataset was produced by pointing at a missing root.
    assert not (missing / "train.csv").exists()


def test_malformed_path_is_a_configuration_error():
    bad = "C:\\definitely\x00not a path"
    try:
        Path(bad).expanduser().resolve()
    except (OSError, ValueError, RuntimeError):
        pass
    else:  # pragma: no cover - platform dependent
        pytest.skip("this platform accepts an embedded NUL in a path")

    args = build_parser().parse_args(["--fundus-root", bad])
    with pytest.raises(ConfigError) as excinfo:
        apply_root_overrides(args)
    message = str(excinfo.value)
    assert "--fundus-root" in message
    assert "Mendeley corpus" in message


def test_kaggle_credentials_are_reported_missing_not_invented(tmp_path, monkeypatch):
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    assert aptos_adapter._kaggle_credentials_present() is False

    layout = aptos_adapter._aptos_folder_present(tmp_path / "empty_aptos")
    assert layout.get("csv_present") is False
    assert layout.get("image_count") == 0


def test_idrid_credentials_are_read_from_the_environment_only(monkeypatch):
    assert idrid_adapter.authenticated_sources()["configured"] is False

    monkeypatch.setenv("DR_IDRID_URLS", "https://example.invalid/a.zip https://example.invalid/b.zip")
    monkeypatch.setenv("DR_IDRID_COOKIE", "session=abc")
    sources = idrid_adapter.authenticated_sources()

    assert sources["configured"] is True
    assert len(sources["urls"]) == 2
    assert sources["cookie"] == "session=abc"


def test_launcher_documents_every_root_flag():
    launcher = (REPO_ROOT / "RUN.bat").read_text(encoding="utf-8", errors="replace")
    for attribute, _env_var, _label in ROOT_OVERRIDE_FLAGS:
        assert _flag(attribute) in launcher, f"RUN.bat does not document {_flag(attribute)}"


def test_mmrdr_root_is_used_by_the_adapter_and_the_inventory(tmp_path, monkeypatch):
    supplied = tmp_path / "MMRDR copy"
    supplied.mkdir()
    (supplied / "MMRDR-CFP").mkdir()
    monkeypatch.setenv("DR_MMRDR_ROOT", str(supplied))

    cfg = load_config()
    assert cfg.mmrdr_root() == supplied.resolve()
    assert dataset_inventory(cfg)["mmrdr"]["root"] == str(supplied.resolve())
