# Wheelhouse

This folder stores only legacy CubeNTube wheel artifacts for downgrade/testing.

- Keep source code in `src/` as latest-only.
- Keep only legacy `.whl` files here (or in GitHub Releases assets).
- Keep the latest build wheel in `dist/` (not in `wheelhouse`).
- Keep wheel filenames in canonical PEP 427 format:
  `chimerax_cubentube-<version>-py3-none-any.whl`
  (do not append custom suffixes to filenames).
- Install a specific version in ChimeraX with:

```bash
toolshed install /absolute/path/to/chimerax_cubentube-<version>-py3-none-any.whl
```

## Artifact Notes

- `chimerax_cubentube-1.0.0-py3-none-any.whl`
  - source: GitHub origin/main snapshot build
  - sha256: `5d0ab68b76f5998b30f76357a53314cf399a1b5425105a9f4e83bfbf58c64d2c`
