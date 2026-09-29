from __future__ import annotations

from pathlib import Path
import shlex

import yaml


REPOSITORIES_ROOT = Path(__file__).resolve().parents[2]
CONTAINER_ROOT = Path(__file__).resolve().parents[1]
APP_ROOTS = {
    "mn-protein-design": REPOSITORIES_ROOT / "mn-protein-design",
    "mn-ligand": REPOSITORIES_ROOT / "ovo-ligand",
}


def test_dockerfiles_do_not_read_host_tool_checkouts() -> None:
    dockerfiles = sorted(
        path
        for root in [CONTAINER_ROOT, *APP_ROOTS.values()]
        for path in root.rglob("Dockerfile*")
    )
    assert dockerfiles
    for dockerfile in dockerfiles:
        assert "tools_to_implement" not in dockerfile.read_text(), dockerfile


def test_app_python_sources_do_not_resolve_host_tool_checkouts() -> None:
    for app_root in APP_ROOTS.values():
        for source_root in (app_root / "mn_protein_design", app_root / "mn_ligand"):
            if not source_root.is_dir():
                continue
            for source in source_root.rglob("*.py"):
                text = source.read_text(errors="replace")
                assert "tools_to_implement" not in text, source
                assert "/home/user/programs/" not in text, source


def test_app_compose_builds_use_shared_container_contexts() -> None:
    for app_name, app_root in APP_ROOTS.items():
        compose_path = app_root / "docker-compose.yml"
        compose = yaml.safe_load(compose_path.read_text())
        for service_name, service in compose["services"].items():
            build = service.get("build")
            if not build:
                continue
            context = (app_root / build["context"]).resolve()
            assert context == CONTAINER_ROOT.resolve(), (app_name, service_name, context)
            assert build.get("additional_contexts", {}).get("app") == ".", (
                app_name,
                service_name,
            )
            assert (CONTAINER_ROOT / build["dockerfile"]).is_file(), (
                app_name,
                service_name,
                build["dockerfile"],
            )


def test_named_app_copy_sources_are_present() -> None:
    for dockerfile in CONTAINER_ROOT.rglob("Dockerfile*"):
        for line in dockerfile.read_text().splitlines():
            if not line.startswith("COPY --from=app "):
                continue
            parts = shlex.split(line.rstrip().removesuffix("\\"))
            source_index = 1
            while parts[source_index].startswith("--"):
                source_index += 1
            source = parts[source_index].lstrip("/")
            app_root = (
                APP_ROOTS["mn-ligand"]
                if dockerfile.is_relative_to(CONTAINER_ROOT / "mn-ligand")
                else APP_ROOTS["mn-protein-design"]
            )
            assert (app_root / source).is_file(), (dockerfile, source)


def test_image_manifest_recipes_exist() -> None:
    manifest = yaml.safe_load((CONTAINER_ROOT / "images.yaml").read_text())
    for tool, image in manifest["images"].items():
        recipe = image.get("recipe")
        if recipe == "external":
            continue
        assert (CONTAINER_ROOT / recipe).is_file(), (tool, recipe)


def test_patches_are_separate_from_tool_source_checkouts() -> None:
    patches = sorted((CONTAINER_ROOT / "patches").glob("*.patch"))
    assert patches
    assert all(path.stat().st_size for path in patches)
