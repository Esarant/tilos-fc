import sys
import json
import os

def convert_notebook_to_script(notebook_path):
    # Check if the file exists
    if not os.path.isfile(notebook_path):
        print(f"File {notebook_path} does not exist.")
        return

    # Open and read the notebook file
    with open(notebook_path, 'r', encoding='utf-8') as nb_file:
        notebook = json.load(nb_file)

    # Prepare the output .py file path
    base_name = os.path.splitext(notebook_path)[0]
    script_path = f"{base_name}.py"

    with open(script_path, 'w', encoding='utf-8') as script_file:
        for cell in notebook.get('cells', []):
            cell_type = cell.get('cell_type')
            source = ''.join(cell.get('source', []))

            if cell_type == 'markdown':
                # Convert markdown cell to commented lines
                script_file.write('\n')
                script_file.write('# ' + '-' * 70 + '\n')
                for line in source.splitlines():
                    script_file.write(f"# {line}\n")
            elif cell_type == 'code':
                # Write code cell content as is
                script_file.write('\n')
                script_file.write('# ' + '=' * 70 + '\n')
                script_file.write(f"{source}\n")

    print(f"Converted {notebook_path} to {script_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_notebook.py <notebook.ipynb>")
    else:
        notebook_file = sys.argv[1]
        convert_notebook_to_script(notebook_file)
