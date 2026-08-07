"""GPU-optimized MCAP converter entrypoint."""

from .convert import main_with_profile


def main(args=None):
    """Run the dedicated GPU-path converter."""
    return main_with_profile(args=args, profile="gpu")


if __name__ == "__main__":
    main()
