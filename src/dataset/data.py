import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split

PATH_1 = r'E:\Data_analysis\NLP_2023.1\NLP.2023.1.Generative-Based-Chatbot\data\dialog_knowledge_data_dialogpt'
PATH_2 = r'E:\Data_analysis\NLP_2023.1\NLP.2023.1.Generative-Based-Chatbot\data\conservation_data'


def create_dataframe(path2):
    dataframes = []
    for disease in os.listdir(path2):
        if '.json' in disease:
            name = disease.replace('.json', '')
            disease_path = os.path.join(path2, disease)
            with open(disease_path, "r") as file1:
                json_data = json.load(file1)

            df = pd.DataFrame(json_data, columns=["response",
                                                  "context 4", 
                                                  "context 3", 
                                                  "context 2", 
                                                  "context 1", 
                                                  "context 0"])
            # df['disease'] = name.replace('_', ' ')    
            dataframes.append(df)
    
    # for data_ in os.listdir(path2):
    #     if '.json' in data_:
    #         data_path = os.path.join(path2, data_)
    #         with open(data_path, "r", encoding='utf-8') as file2:
    #             json_data = json.load(file2)

    #         data = []
    #         # Iterate through each JSON object and append to the data list
    #         for obj in json_data:
    #             row = {}
    #             if 'response' in obj:
    #                 row['response'] = obj['response']
    #             else:
    #                 row['response'] = None
    #             for i in range(8, -1, -1):
    #                 context_key = f'context{i}'
    #                 if context_key in obj:
    #                     row[context_key] = obj[context_key]
    #                 else:
    #                     row[context_key] = None
    #             data.append(row)
    #         # Create the DataFrame using the list of dictionaries
    #         df = pd.DataFrame(data)
    #         # Reorder columns based on the desired order
    #         column_order = ['response'] + [f'context{i}' for i in range(8, -1, -1)]
    #         df = df[column_order]
    #         dataframes.append(df)
    
    result_df = pd.concat(dataframes, ignore_index=True)
    result_df.to_csv(r'E:\Data_analysis\NLP_2023.1\NLP.2023.1.Generative-Based-Chatbot\data\df.csv')
    return result_df

                    

def train_val_split():
    df = create_dataframe(PATH_1)
    trn_df, val_df = train_test_split(df, test_size=0.1)
    trn_df.to_csv(r'E:\Data_analysis\NLP_2023.1\NLP.2023.1.Generative-Based-Chatbot\data\train_.csv')
    val_df.to_csv(r'E:\Data_analysis\NLP_2023.1\NLP.2023.1.Generative-Based-Chatbot\data\val_.csv')


if __name__ == "__main__":
    df = create_dataframe(PATH_2)