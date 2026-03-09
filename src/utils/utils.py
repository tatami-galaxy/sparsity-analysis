import os
from os.path import dirname


def get_root_dir():
    root = os.path.abspath('')
    project_name = 'sparsity-analysis'
    #print("Project name set as {}. Make sure it is correct".format(project_name))
    while root.split('/')[-1] != project_name:
        root = dirname(root)
    return root