import os

ARTICLES = 1 # number of related articles 
INTERVAL = 1.2 # time interval to access new URLs 
OUTPUT = '../raw_data'
file_path = './context_url.txt'
disease_path = './disease.txt'
if __name__ == "__main__": 

    def main(url, context_folder):
        pass 

    with open(disease_path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            disease = line.replace(' ', '_').replace('\n', '')
            url = 'https://en.wikipedia.org/wiki/' + disease
            disease_folder = os.path.join(OUTPUT, disease)

            if os.path.exists(disease_folder) == False:
                os.mkdir(disease_folder)
            
            run = 'python spider.py {0} --output={1} --articles={2} --interval={3}'.format(url, disease_folder, ARTICLES, INTERVAL)
            os.system(run)

