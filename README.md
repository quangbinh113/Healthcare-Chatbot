# NLP.2023.1.Generative-Based-Chatbot

The objective is to create a HealthCare-Domain chatbot that can generate answers itself.
Our Project focuses on various domains (Simple chitchat, Question & Answering, Disease Prediction, etc..)

# For Data

In the ./data_crawler/data/src/context_url.txt paste some specific links of the topic. 
Next, go to run_spider.py and run. It will access the topic url and retrieve all the strings within it, and perform the same process with 100 links contained within the topic link. Finally, save them to separate files within a folder named after the topic of the original link. 

We aim to use ChatGPT-API to generate Questions and Answering from the contexts scrapted from the webs.

Our dataset will contain several features (Question, Answer, Context, Dialog, intent_tag, action_tag).
