from pathlib import Path

OUTPUT_FILE = "project_dump.txt"

INCLUDE_EXTENSIONS = {
    ".py",
    ".yml",
    ".yaml",
    ".json",
    ".toml",
    ".txt",
    ".md",
    ".dockerignore",
    ".gitignore",
    ".ini",
    ".cfg",
}

INCLUDE_FILENAMES = {
    "Dockerfile",
    "docker-compose.yml",
    "requirements.txt",
    "Makefile",
    ".env.example",
}

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    "data",
    "mlruns",
    "artifacts",
    "checkpoints",
    "models",
    "dist",
    "build",
}

EXCLUDE_EXTENSIONS = {
    ".pt",
    ".pth",
    ".ckpt",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".zip",
    ".tar",
    ".gz",
    ".csv",
    ".parquet",
    ".db",
    ".sqlite",
}

root = Path.cwd()

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    out.write(f"PROJECT DUMP: {root.name}\n")
    out.write("=" * 100 + "\n\n")

    for path in sorted(root.rglob("*")):

        # Skip directories
        if path.is_dir():
            continue

        # Skip excluded directories
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue

        # Skip excluded extensions
        if path.suffix.lower() in EXCLUDE_EXTENSIONS:
            continue

        # Check inclusion
        should_include = (
            path.suffix.lower() in INCLUDE_EXTENSIONS
            or path.name in INCLUDE_FILENAMES
        )

        if not should_include:
            continue

        relative_path = path.relative_to(root)

        out.write("\n")
        out.write("=" * 100 + "\n")
        out.write(f"FILE: {relative_path}\n")
        out.write("=" * 100 + "\n\n")

        try:
            content = path.read_text(encoding="utf-8")
            out.write(content)
        except Exception as e:
            out.write(f"[Could not read file: {e}]")

        out.write("\n\n")

print(f"\nProject dump created: {OUTPUT_FILE}")
