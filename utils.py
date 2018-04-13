import numpy as np
import os

def join_line(data, sep=" "):
    '''
    Function that converts array into string

    Parameters:
    - data: array-like
    - sep: string

    Return: string
    '''
    return sep.join(map(str, data))

def file_append(path, data, sep=" ", newline=True):
    '''
    Function that adds line to a file with joined data

    Parameters:
    - path: string
    - data: array-like
    - sep: string

    Return: void
    '''
    f = open(path, "a")
    output = join_line(data, sep)
    if(newline): output += "\n"
    f.write(output)
    f.close()

def read_columns_as_rows(path, columns, sep='\t', format=float):
    '''
    Function that reads selected columns from a file and returns them as rows

    Parameters:
    - path: string
    - columns: list of integeres
    - sep: string
    - format: array type

    Return: numpy array
    '''
    res = []
    if os.path.isfile(path):
        with open(path, 'r') as file:
            for line in file:
                elements = line.split(sep)
                res.append([elements[i] for i in columns])
            res = np.asarray(res).transpose().astype(format)
    return res
# Note: in the approach below the whole file is read 
# but is more elastic i.e. columns may be list or integer
#         res.append(line.split(sep))
# return np.asarray(res)[:, columns].astype(format)