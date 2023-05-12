import os

ARTICLES = 10
INTERVAL = 1.2
OUTPUT = 'E:/Data_analysis/NLP_chatbot/Data'

if __name__ == "__main__":
    file_path = r"E:\Data_analysis\NLP_chatbot\context_url.txt"  # Replace with the actual file path

    # Open the file in read mode
    with open(file_path, "r") as file:
        # Read all the lines from the file
        lines = file.readlines()

    # Print each line
    for line in lines:
        print(line.strip())  # Use .strip() to remove leading/trailing whitespace and newline characters


    run = './src/spider.py https://en.wikipedia.org/wiki/Diabetes --output={0} --articles={1} --interval={2}'.format(OUTPUT, ARTICLES, INTERVAL)
    os.system(run)

