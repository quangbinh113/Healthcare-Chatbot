# Health-care Chatbot

The objective is to create a HealthCare-Domain Generative-based chatbot that can provide valuable information about health issues and help people to figure out how to prevent themself from diseases.

# Requirement 

In your cmd run this command: 'pip install -r requirement.txt' to download all packets needed.

# The data for training Health-care Chatbot 

You can generate the data by visit the `data` folder under the `root` folder

- **Folder `get_data`**: Contains the python file to crawl data from Wikipedia and then transform it into dialogues by the power of OpenAI API.

# We propose two versions of chatbot using two different models

## DialoGPT
- **Folder `DialoGPT`**: Contains the source codes to train the bot using GPT-2 based model with the data consists of seven contexts for each dialouge.
## Godel
- **Folder `Godel`**: Contains the source codes to train the bot using Godel model with the data consists of 11 contexts associated with its knowledge(context) for each dialouge.

# How to use?
In  **Folder `views`** you can see a "Chatbot.py" file. Open your cmd and run command: "streamlit run Chatbot.py"

![alt text](https://github.com/quangbinh113/healthcare-chatbot/blob/main/data/images/medi.png?raw=true)

# Lisense
GPL-3.0 License

# Contributors

- [Truong Quang Binh](https://github.com/quangbinh113)
- [Le Tran Thang](https://github.com/thang662)
- [Tran Khanh Luong](https://github.com/collaborator2)
- [Ly Nhat Nam](https://github.com/collaborator2)

