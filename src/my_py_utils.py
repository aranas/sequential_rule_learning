'''some handy functions'''

def list2flat(alist):
    '''flattens a nested list'''
    new_list = []
    for item in alist:
        if isinstance(item, list):
            for subitem in item:
                new_list.append(subitem)
        else:
            new_list.append(item)

    return new_list
