import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split

PATH = r'NLP.2023.1.Generative-Based-Chatbot\data\conservation_data'


def create_dataframe(path):
    dataframes = []
    for disease in os.listdir(path):
        if '.json' in disease:
            name = disease.replace('.json', '')
            disease_path = os.path.join(path, disease)
            with open(disease_path, "r") as file:
                json_data = json.load(file)

            df = pd.DataFrame(json_data, columns=["response",
                                                  "context 4", 
                                                  "context 3", 
                                                  "context 2", 
                                                  "context 1", 
                                                  "context 0"])
            df['disease'] = name
            dataframes.append(df)

    result_df = pd.concat(dataframes, ignore_index=True)
    result_df.to_csv(r'NLP.2023.1.Generative-Based-Chatbot\data\df.csv')
    return result_df

                    

def train_val_split():
    df = create_dataframe(PATH)
    y = df['disease']
    x = df.drop(columns='disease')
    trn_df, val_df, y_train, y_val = train_test_split(x, y, test_size = 0.1)  
    trn_df = pd.concat([trn_df, y_train], axis=1) # Add y_train to x_train
    val_df = pd.concat([val_df, y_val], axis=1)    # Add y_test to x_test
    # trn_df = pd.DataFrame(x_train)
    # val_df = pd.DataFrame(x_val)
    trn_df.to_csv(r'NLP.2023.1.Generative-Based-Chatbot\data\train.csv')
    val_df.to_csv(r'NLP.2023.1.Generative-Based-Chatbot\data\val.csv')


if __name__ == "__main__":
    train_val_split()