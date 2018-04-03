from pickle import dump
import os
import string
path = "/media/prachi/New Volume/3-2/LOP/Dataset/cnn_stories_tokenized/"

def separate(doc):
    index=doc.find('@highlight') 
    #creating story and list of corresponding highlights
    story, highlights = doc[:index], doc[:index].split('@highlight') 
    highlights = [i.strip() for i in highlights] 
    return story, highlights

def load_file(path):
    stories=list()
    for story in os.listdir(path):
        story_name=path+story
        story_name=open(story_name,encoding='utf-8')
        doc=story_name.read()
        story_name.close()
        story, highlights = separate(doc)
        stories.append({'s':story , 'h':highlights})
    return stories
        
def clean_data(lines):
    new_lines = list()
    #If three arguments are passed, each character in the third argument is mapped to None
    table = str.maketrans('', '', string.punctuation)
    for line in lines:
        index = line.find('(CNN) -- ')
        if index > -1:
            line = line[index+len('(CNN)'):]
        line = line.split()
        line = [word.lower() for word in line]
        line = [w.translate(table) for w in line]
        line = [word for word in line if word.isalpha()] 
        new_lines.append(' '.join(line))
    new_lines = [l for l in new_lines if l]    
    return new_lines        
 

stories=load_file(path)
for story in stories:
    story['s'] = clean_data(story['s'].split('\n'))
    story['h'] = clean_data(story['h']) 
    
dump(stories, open('cnn_dataset.pkl', 'wb+')) 
