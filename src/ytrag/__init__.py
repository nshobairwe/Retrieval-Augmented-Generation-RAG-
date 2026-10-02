def main() -> None:
    print("Hello from ytrag!")

from .frontend import main as frontend_main
__all__ = ["frontend_main"]
