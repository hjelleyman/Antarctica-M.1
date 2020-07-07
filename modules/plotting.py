import os
size = os.get_terminal_size()

def print_heading(heading):
    """Summary
    
    Parameters
    ----------
    heading : TYPE
        Description
    """
    sidespace = size[0]//4
    if len(heading)%2==1:
        heading = ' '+heading
    print('-'*((size[0]-2*sidespace)//2 * 2))
    print(' '*((size[0]-2*sidespace-2-len(heading))//2)+heading+' '*((size[0]-2*sidespace-2-len(heading))//2)+' ')
    print('-'*((size[0]-2*sidespace)//2 * 2))