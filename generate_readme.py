import os

def generate_project_structure(path='.', level=0):
    project_structure = ''
    for root, dirs, files in os.walk(path):
        indent = ' ' * 4 * (root.count(os.sep) - level)
        project_structure += f'{indent}{os.path.basename(root)}/\n'
        for file in files:
            project_structure += f'{indent}    {file}\n'
        break  # Remove this line if you want a full depth scan
    return project_structure

if __name__ == "__main__":
    structure = generate_project_structure()
    with open('README.md', 'a') as readme:
        readme.write("\n## Structure du Projet\n\n")
        readme.write("```\n")
        readme.write(structure)
        readme.write("```\n")
