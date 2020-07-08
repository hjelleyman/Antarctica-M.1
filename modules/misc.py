import numpy as np
import os
import matplotlib.pyplot as plt
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




def seaice_area_mean(seaice):
    x = seaice.x.values.copy()
    y = seaice.y.values.copy()
    x, y = np.meshgrid(x,y)

    dx = np.diff(x)[0,0]
    dy = np.diff(y)[0,0]
    dA = 4 * dx * dy / (1 + x**2 + y**2)**2

    weighted_seaice = seaice*dA
    weighted_mean = weighted_seaice.mean(dim=['x','y'])

    plt.plot(weighted_mean)
    plt.show()
    print(weighted_mean)
    return weighted_mean