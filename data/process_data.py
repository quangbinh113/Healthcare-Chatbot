import os
import json


def preprocess(source_path):
    for filename in os.listdir(source_path):
        # Read the JSON file
        with open(os.path.join(source_path, filename), 'r') as file:
            data = json.load(file)

            # Iterate through each JSON object in the file
            for i, obj in enumerate(data):
                # Perform your modifications here
                # For example, adding additional entries to the "dialog" list
                if len(obj['dialog']) < 9:
                    new_dialog = [""] * (9 - len(obj['dialog'])) + obj['dialog']
                    data[i]['dialog'] = new_dialog

            # Write the updated data back to the JSON file
            with open(os.path.join("./processed_data", filename), 'w') as file:
                json.dump(data, file, indent=4)


def format_json(source_path):
    # format all json file in the source_path
    filenames = os.listdir(source_path)
    for filename in filenames:
        if filename.endswith('.json'):
            path = os.path.join(source_path, filename)
            with open(path, 'r') as f:
                data = json.load(f)
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)

if __name__ == "__main__":
    source_path = './final_data'
    preprocess(source_path)
    format_json(source_path)
    format_json('./processed_data')
   