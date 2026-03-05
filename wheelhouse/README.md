# Wheelhouse

This folder stores CubeNTube wheel artifacts for easy downgrade/testing.

- Keep source code in `src/` as latest-only.
- Keep old `.whl` files here (or in GitHub Releases assets).
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
- `chimerax_cubentube-1.1.0-py3-none-any.whl`
  - source: latest local working tree clean build
  - sha256: `bc9b90b4631a42dbdd060eb82ee2001a8854855951006132e2f14a76235f0ada`
