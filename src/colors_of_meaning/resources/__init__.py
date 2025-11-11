from pathlib import Path


def get_resource_path(resource_name: str) -> str:
    base_dir = Path(__file__).parent
    resource_path = base_dir / resource_name

    if not resource_path.exists():
        raise FileNotFoundError(f"Resource {resource_name} not found at {resource_path}")

    return str(resource_path)
