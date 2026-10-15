"""Run the FedForge smoke simulator when the package is executed directly."""

from .simulator import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
