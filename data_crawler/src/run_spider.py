import os

ARTICLES = 50
INTERVAL = 1.2
OUTPUT = './NLP.2023.1.Generative-Based-Chatbot/data_crawler/data'
file_path = './NLP.2023.1.Generative-Based-Chatbot/data_crawler/src/contect_url.txt'


if __name__ == "__main__":  

    with open(file_path, "r") as file:
        lines = file.readlines()

        for line in lines:
            url = line.replace('\n', '')
            context = url.replace('https://en.wikipedia.org/wiki/', '')
            context_folder = os.path.join(OUTPUT, context)

            if os.path.exists(context_folder) == False:
                os.mkdir(context_folder)

            run = 'python ./NLP.2023.1.Generative-Based-Chatbot/data_crawler/src/spider.py {0} --output={1} --articles={2} --interval={3}'.format(url, context_folder, ARTICLES, INTERVAL)
            os.system(run)
            # print

