import os
def processing_text(text):
    text = text.replace("\\xc2\\xa0","").replace("\\xe2\\x80\\x93","").replace("\\xe2\\x82\\xac27\\","").replace("\\xe2\\x80\\x94","").replace("\\xe2\\x89\\xa5","").replace("\\xe2\\x82\\xac","")
    return text

def chunking(context,n):
    context = processing_text(context)
    context_list = context.split('.')
    return [". ".join(context_list[i:i+n]) for i in range(0,len(context_list),n)]
if __name__ == "__main__":
    root = os.getcwd()
    
    total = 0
    for name in os.listdir(os.path.join(root,"raw_data")):
        with open(os.path.join(root,"raw_data",name,name+".txt"),"r")as f:
            h = f.read()
        data_chunk = chunking(h, 7)
        for i,c in enumerate(data_chunk):
            with open(os.path.join(root,"chunking",name+"_"+str(i)+".txt"),"w")as q:
                q.write(name.replace("_"," ")+" "+name.replace("_"," ")+" "+c+" "+name.replace("_"," ")+" "+name.replace("_"," "))
        total += len(data_chunk)
    print(total)



