def join_line(data, separator=" "):
    '''
    Function that joins data with a given separator

    Parameters:
    - data: array-like
    - separator: string
    '''
    return separator.join(map(str, data))

def file_append(filename, data, separator=" ", newline=True):
    '''
    Function that adds line to file with joined data

    Parameters:
    - filename: string
    - data: array-like
    - separator: string
    '''
    f = open(filename, "a")
    output = join_line(data, separator)
    if(newline): output += "\n"
    f.write(output)
    f.close()