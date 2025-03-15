import os
import re
import json

input_dir = "gesetze/"
cleaned_output_dir = "cleaned_gesetze/"
output_json = "extracted_metadata.json"

os.makedirs(cleaned_output_dir, exist_ok=True)

data = []

def extract_metadata_and_content(filepath):
    """Extracts metadata and content from a Markdown file."""
    metadata = {}
    content_lines = []
    in_metadata = False

    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line == "---":
                in_metadata = not in_metadata
                continue
            if in_metadata:
                match = re.match(r"^([\w-]+):\s*(.+)$", line)
                if match:
                    key, value = match.groups()
                    metadata[key] = value
            else:
                content_lines.append(line)

    content = "\n".join(content_lines)
    return metadata, content

def clean_text(content):
    """Cleans the text content."""
    content = re.sub(r'\n{2,}', '\n', content)
    content = re.sub(r'(\s+):\s+', ': ', content)
    return content.strip()

for filename in os.listdir(input_dir):
    if filename.endswith(".md"):
        filepath = os.path.join(input_dir, filename)
        metadata, content = extract_metadata_and_content(filepath)

        cleaned_content = clean_text(content)
        cleaned_filename = f"{os.path.splitext(filename)[0]}_cleaned.md"
        cleaned_filepath = os.path.join(cleaned_output_dir, cleaned_filename)
        with open(cleaned_filepath, 'w', encoding='utf-8') as cleaned_file:
            cleaned_file.write(cleaned_content)

        metadata["filename"] = filename
        metadata["cleaned_content_path"] = cleaned_filepath
        data.append(metadata)

with open(output_json, 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)

print(f"Metadata and cleaned content paths saved to {output_json}")
print(f"Cleaned content saved in {cleaned_output_dir}")
