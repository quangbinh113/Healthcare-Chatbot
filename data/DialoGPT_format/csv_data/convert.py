import json
import os
import pandas as pd

def convert_to_csv(source_path):
     for filename in os.listdir(source_path):
        # Read the JSON file
        with open(os.path.join(source_path, filename), 'r') as file:
            data = json.load(file)
        # Initialize an empty list to store the rows of the dataframe
        rows = []
        filename = filename.replace('.json', '.csv')
        # Iterate through each JSON object in the file
        for obj in data:
            # Extract the relevant values from the JSON object
            respond = obj['response']
            context = obj['dialog']  # Reverse the order of dialog elements
            
            # Create a dictionary with the values for each column
            row = {
                'respond': respond,
                'context_8': context[8] ,
                'context_7': context[7] ,
                'context_6': context[6] ,
                'context_5': context[5] ,
                'context_4': context[4] ,
                'context_3': context[3] ,
                'context_2': context[2] ,
                'context_1': context[1] ,
                'context_0': context[0] 
            }
            
            # Append the row to the list
            rows.append(row)

        # Create the dataframe from the list of rows
        df = pd.DataFrame(rows)
        # Replace the empty strings with special value
        df = df.replace(r'^\s*$', 'EMPTY', regex=True)
        # count the number of row has NA value
        count = df.isna().sum().sum()
        # Save the dataframe to CSV format
        df.to_csv(f"./{filename}", index=False)


                    



if __name__ == "__main__":
    source_path = '../processed_data'
    convert_to_csv(source_path)