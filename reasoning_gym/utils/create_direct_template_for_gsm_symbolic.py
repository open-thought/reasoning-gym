import os
import json
import re

def convert_to_direct_template(question_annotated):
    """
    Convert question_annotated from explicit mapping format to direct template format.

    Args:
        question_annotated (str): The original annotated question string.

    Returns:
        str: The modified question string in direct template format.
    """
    return re.sub(r"\{(\w+),[^}]+\}", r"{\1}", question_annotated)

def process_json_files_in_folder(src_folder_path, dest_folder_path):
    """
    Load all JSON files in a folder, process the 'question_annotated' field,
    and save the modified JSON files back.

    Args:
        folder_path (str): The path to the folder containing JSON files.
    """
    # Ensure the folder exists
    if not os.path.isdir(src_folder_path):
        print(f"Error: Folder '{src_folder_path}' not found.")
        return

    # Loop through each file in the src folder
    for filename in os.listdir(src_folder_path):
        if filename.endswith(".json"):  # Process only JSON files
            file_path_to_load = os.path.join(src_folder_path, filename)
            file_path_to_save = os.path.join(dest_folder_path, filename)
            print(f"Processing: {filename}")
            print(f"File path to save: {file_path_to_save}")
            # Read the JSON file
            try:
                with open(file_path_to_load, "r", encoding="utf-8") as file:
                    data = json.load(file)

                # Check if 'question_annotated' exists and modify it
                if "question_annotated" in data:
                    data["question_annotated"] = convert_to_direct_template(data["question_annotated"])

                    # Save the modified JSON back to the file
                    with open(file_path_to_save, "w", encoding="utf-8") as file:
                        json.dump(data, file, indent=4)

                    print(f"Processed: {filename}")

            except Exception as e:
                print(f"Error processing '{filename}': {e}")

# Example usage
src_folder_path = "./reasoning_gym/data/gsm_data/symbolic"  # Replace with actual folder path
dest_folder_path = "./reasoning_gym/data/gsm_data/direct_template_symbolic"  # Replace with actual folder path

if __name__ == "__main__":
    process_json_files_in_folder(src_folder_path=src_folder_path, dest_folder_path=dest_folder_path)

