import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.utils.server_cli import show_interactive_server_commands
from app.tools.microscopy import take_image, close_microscope


def run_manual_cli_take_image_test() -> None:
    """
    Manual (non-pytest) test flow:
    1) User starts servers + connects through interactive CLI prompts.
    2) User triggers image acquisition.
    3) Script reports saved image path and tears down context.
    """
    print("=" * 60)
    print("Manual Interactive CLI Test: Start/Connect then Take Image")
    print("=" * 60)
    print("Follow the prompts to configure and start the microscope context.")

    image_path = None
    try:
        show_interactive_server_commands()

        detector = input("\nDetector name [haadf]: ").strip() or "haadf"
        result = take_image(detector=detector)

        print("\n[TAKE IMAGE RESULT]")
        print(result)

        match = re.search(r"saved to (.+?) \(Shape", result)
        if match:
            image_path = match.group(1)
            print(f"\nImage saved to: {image_path}")
        else:
            print("\nCould not parse an output image path from the result.")
    finally:
        close_result = close_microscope()
        print("\n[CLEANUP]")
        print(close_result)

        if image_path and os.path.exists(image_path):
            cleanup_choice = input("Delete captured image file? [y/n]: ").strip().lower()
            if cleanup_choice in ("", "y", "yes"):
                os.remove(image_path)
                print(f"Deleted: {image_path}")


if __name__ == "__main__":
    run_manual_cli_take_image_test()
