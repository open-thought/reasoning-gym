import importlib.util
import os
from datetime import datetime


def test_generator_files(directory_path: str) -> None:
    """Test all generator files in directory"""

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"generator_tests_{timestamp}.log"

    with open(log_file, "w") as log:
        # Walk through directory
        for root, _, files in os.walk(directory_path):
            for file in files:
                if not file.endswith(".py"):
                    continue

                filepath = os.path.join(root, file)
                try:
                    # Import module dynamically
                    spec = importlib.util.spec_from_file_location(file[:-3], filepath)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Check for original_example function
                    if hasattr(module, "original_example"):
                        try:
                            result = module.original_example()
                            if isinstance(result, dict):
                                message = f"SUCCESS: {file} - returned valid dictionary\n"
                            else:
                                message = f"ERROR: {file} - did not return dictionary\n"
                        except Exception as e:
                            message = f"ERROR: {file} - execution failed: {str(e)}\n"
                    else:
                        message = f"ERROR: {file} - no original_example() found\n"

                except Exception as e:
                    message = f"ERROR: {file} - import failed: {str(e)}\n"

                # Log result
                log.write(message)
                print(message, end="")


# Usage
if __name__ == "__main__":
    generator_path = "../reasoning_gym/arithmetic/gsm_symbolic"
    test_generator_files(generator_path)
