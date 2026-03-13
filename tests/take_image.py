import os
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from app.tools.microscopy import get_eds_spectrum, close_microscope, connect_client, start_server, take_image


def _require_success(result: str, expected_substrings: tuple[str, ...], step_name: str) -> str:
	if not any(substring in result for substring in expected_substrings):
		raise RuntimeError(f"{step_name} failed: {result}")
	return result


def main() -> int:
	image_path = None

	try:
		start_result = start_server(mode="real")
		print(start_result)
		_require_success(
			start_result,
			(
				"Started PyTango asyncroscopy servers",
				"PyTango asyncroscopy servers already running",
			),
			"Server startup",
		)

		connect_result = connect_client()
		print(connect_result)
		_require_success(
			connect_result,
			("Connected to PyTango microscope",),
			"Microscope connection",
		)

		image_result = take_image(detector="haadf")
		print(image_result)
		_require_success(
			image_result,
			("Image captured from detector 'haadf'",),
			"Image capture",
		)

		match = re.search(r"saved to (.+?) \(Shape", image_result)
		if not match:
			raise RuntimeError(f"Could not parse image path from result: {image_result}")

		image_path = match.group(1)
		if not os.path.exists(image_path):
			raise RuntimeError(f"Image file was not created: {image_path}")

		print(f"Saved image: {image_path}")

		res = get_eds_spectrum()
		print(f"EDS result: {res}")	

		return 0
	except Exception as exc:
		print(f"Error: {exc}", file=sys.stderr)
		return 1
	
	finally:
		close_result = close_microscope()
		print(close_result)
		if image_path and os.path.exists(image_path):
			os.remove(image_path)
			print(f"Removed image: {image_path}")


if __name__ == "__main__":
	raise SystemExit(main())