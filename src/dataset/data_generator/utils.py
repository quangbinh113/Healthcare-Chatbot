import os
import numpy as np
import json

LIMIT_WORDS = 600

def count_words(context):
    return len(context.split(' '))

def split_to_contexts(file_path):
    # read txt file 
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # split the text into contexts, each context no more than 500 words
    contexts = []
    sentences = text.split('.')
    context = ''
    i = 0 
    while i < len(sentences):
        context += sentences[i] + '.'
        if count_words(context) > LIMIT_WORDS:
            contexts.append(context)
            context = ''
        i += 1 
        
    return contexts 

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

def update_data(json_strings, file_path):
    for json_string in json_strings:
        # Read the existing JSON file and parse its contents
        with open(file_path, "r") as file:
            old_data = json.load(file)

        # print("Response:", json_string)
        # Convert the JSON string to a list of dictionaries
        try:
            updated_data = json.loads(json_string)
            # Extend the existing data with the new data
            merged_data = old_data + updated_data

            # Write the merged data back to the JSON file
            with open(file_path, "w") as file:
                json.dump(merged_data, file)
        except Exception as e:
            print('Cannot update data, call the function again')

def count_conversations(source):
    sum = 0
    filenames = os.listdir(source)
    for filename in filenames:
        file_path = os.path.join(source, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
        sum += len(data)
    return sum
if __name__ == '__main__':
    print("***Information about data***")
    total_contexts = []
    base_path = '../../data/raw_data'
    diseases = os.listdir(base_path) 
    for d in diseases:
        file_path = os.path.join(base_path, d, f'{d}.txt')
        contexts = split_to_contexts(file_path)
        if len(contexts) >= 10:
            contexts = contexts[:10]
        total_contexts += contexts
        # print(f"{d} has {len(contexts)} contexts")
    print("The number of contexts from raw data", len(total_contexts))
    # print("A random sample: \n", total_contexts[np.random.randint(len(total_contexts))])
    print("The number of generated conversations from raw_data: ", count_conversations('../../data/conservation_data'))
    # with open('../../data/conservation_data/Gastritis.json', 'r') as f:
    #     data = json.load(f)
    #     print(len(data))
    
    
        
    
        
        
    