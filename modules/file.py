"""
File module: Read, write, edit files.
"""
def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)

def append_file(path, content):
    with open(path, 'a') as f:
        f.write(content)
