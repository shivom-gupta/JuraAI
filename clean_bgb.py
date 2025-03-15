import os
from langchain.text_splitter import MarkdownHeaderTextSplitter
import re

input_folder = "data/bgb"
output_folder = "data/cleaned_bgb"

os.makedirs(output_folder, exist_ok=True)

def process_section(section, skip_texts):
    metadata = section.get("metadata", {})
    page_content = section.get("page_content", "")
    
    if not page_content:
        return None
    
    if 'Header 1' in metadata:
        if len(metadata) == 1:
            return f"# {metadata['Header 1']}\n{page_content}"
        elif 'Header 2' in metadata:
            header_2 = metadata['Header 2']
            if header_2 in ['Artikelübersicht', 'Für den Rechtsverkehr', 'Expertenhinweise']:
                if any(skip_text.lower() in page_content.lower().strip() for skip_text in skip_texts):
                    print(f"Skipping: {header_2}")
                    return None
                if 'Header 4' in metadata:
                    return f"#### {metadata['Header 4']}\n{page_content}"
                elif metadata.get('Header 3') not in ['Fußnoten', 'Anlage', 'Anlagen', 'Anhang', 'Anhänge']:
                    return f"### {metadata['Header 3']}\n{page_content}"
                return f"## {header_2}\n{page_content}"
    return None

def extract_useful_text(content):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ('####', 'Header 4'),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

    content = re.sub(r'http[s]?://\S+', '', content)
    content = re.sub(r'<[^>]+>', '', content)
    sections = markdown_splitter.split_text(content)
    skip_texts = ['dieser Norm ist noch keine Kommentierung veröffentlicht'.lower().strip(), 'für den Rechtsverkehr, häufige Anwendungsfälle'.lower().strip()]

    useful_sections = []
    for section in sections:
        text = process_section(dict(section), skip_texts)
        if text:
            useful_sections.append(text)


        return "\n".join(useful_sections)

for file_name in os.listdir(input_folder):
    if "weggefallen" in file_name:
        continue

    input_path = os.path.join(input_folder, file_name)

    if os.path.isfile(input_path):
        with open(input_path, 'r', encoding='utf-8') as file:
            content = file.read()

        cleaned_text = extract_useful_text(content)
        if cleaned_text == "":
            continue

        output_path = os.path.join(output_folder, file_name)
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(cleaned_text)

print(f"Cleaned files saved to: {output_folder}")
