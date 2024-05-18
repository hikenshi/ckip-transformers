import os
import random
import pandas as pd
import re
from ckip_transformers.nlp import CkipNerChunker

def process_text(text, ner_driver):
    text = text.strip()
    if not text:
        return None
    ner = ner_driver([text])
    persons = [entity.word for entity in ner[0] if entity.ner == 'PERSON' and len(entity.word) >= 2]
    if persons:
        return persons
    return None

def search_chinese(names_data, query_text):
    for zh, vi in names_data:
        if query_text.strip() == zh.strip():
            return vi
    return None  # Return None if no match is found

def load_names_data(file_path):
    names_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        next(file)  # Skip the header line
        for line in file:
            line = line.strip()
            if ',' in line:
                # Split the line using comma delimiter
                parts = line.split(',')
            elif '=' in line:
                # Split the line using equal sign delimiter
                parts = line.split('=')
            else:
                # Ignore lines that don't have the expected format
                continue

            if len(parts) >= 2:
                names_data.append((parts[0], parts[1]))

    return names_data

def process_files(filepath, names_data_file, ner_driver):
    names_data = load_names_data(names_data_file)
    processed_names = set()  # Set to store processed names globally
    with open(filepath, 'r', encoding='utf-8') as FileIn, open('out.txt', 'w', encoding='utf-8') as outfile:
        for line in FileIn:
            result = process_text(line.strip(), ner_driver)
            if result:
                unique_names = []
                vi_text = []
                for name in result:
                    if len(name) < 2 or name in processed_names:
                        continue  # Skip already processed names
                    processed_names.add(name)  # Add the name to the global set of processed names
                    unique_names.append(name)
                    value_text = search_chinese(names_data, name)
                    if value_text is None:
                        vi_text.append("NONE")
                    else:
                        value_text = value_text.title()
                        vi_text.append(value_text)

                if unique_names:  # Only write to the file if there are unique names
                    outfile.write(str(unique_names) + "||" + str(vi_text) + "\n")

if __name__ == "__main__":
    filepath = './一胎俩宝，老婆大人别想逃_utf8.txt'  # path to your directory containing text files
    names_data_file = 'Names.txt'
    ner_driver = CkipNerChunker(model="bert-base")  # Initialize ner_driver in __main__
    process_files(filepath, names_data_file, ner_driver)
