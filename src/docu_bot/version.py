import os
import re

def get_version():
    # Get the version from the CHANGELOG.md file
    
    def search_version(text):
        match = re.search(r'## \[(\d+\.\d+\.\d+)\]', text)
        if match:
            return match.group(1)
        return None
    
    filename = os.path.join(os.path.dirname(__file__), 'CHANGELOG.md')
    with open(filename, 'r') as f:
        text = f.read()
    return search_version(text)

__version__ = get_version()