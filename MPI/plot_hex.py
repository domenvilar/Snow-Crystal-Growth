import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import matplotlib
import numpy as np
from numba import njit
import time
import os
import struct


def load_from_bin(filename:str,n:int,m:int,data_type='B'):
    '''loads an array from file filename of dimensions n x m
    :param filename: the file name
    :param n: the number of matrix rows
    :param m: the number of columns
    :param data_type: the data type to load (B for unsigned char, f for float)'''
    file_size = os.path.getsize(filename)

    element_size = struct.calcsize(data_type)
    num_elements = file_size//element_size

    with open(filename,'rb') as f:
        binary_data = f.read()
        char_array = struct.unpack(f'{num_elements}{data_type}',binary_data)

    arr = np.array(char_array)

    arr = arr.reshape((n,m))

    return arr


SQ3 = np.sqrt(3)

NEIGHS_EVEN_COL = np.array([[-1,0],[-1,1],[0,1],[1,0],[0,-1],[-1,-1]])
NEIGHS_ODD_COL = np.array([[-1,0],[0,1],[1,1],[1,0],[1,-1],[0,-1]])

@njit
def simulate_step(levels1:np.ndarray,levels2:np.ndarray,mat1:np.ndarray,mat2:np.ndarray,alpha:float,gamma:float):
    '''advances the simulation for one step, takes the data from levels1 and mat2 and updates mat2 and levels2
    :param levels1: the input water levels matrix
    :param levels2: the output water levelsm matrix
    :param mat1: input cell flags
    :param mat2: output cell flags
    :param alpha: model parameter associated with diffusion
    :param gamma: model parameter associated with outside vapor
    '''

    alpha12 = alpha/12
    alpha2 = alpha/2
    update_flag = 0

    for i in range(1,levels1.shape[0]-1): # loop over non-edge cells
        for j in range(1,levels1.shape[1]-1):

            if mat1[i,j] == 3: continue # if the cell is frozen, skip the step
            
            # add the inflow from the neighbor cells
            col = NEIGHS_EVEN_COL if j%2 == 0 else NEIGHS_ODD_COL # pick the correct neighbor indexation

            for k in range(6): # loop over all neighbors

                ni,nj = i + col[k,0],j + col[k,1] # find the neighbor indices
                # no water comes from neighbor receptive (2,3) cells
                if mat1[ni,nj] < 2: levels2[i,j] += alpha12 * levels1[ni,nj] 
            
            if mat1[i,j] == 2: # add the outside vapour if the cell is boundary
                levels2[i,j] += gamma

                if levels2[i,j] >= 1: # if the cell freezes, update the neighbor cells
                    update_flag = 1
                
                    mat2[i,j] = 3 # update the flag of the cell
                
                    for k in range(6): # update the neighbor cells
                        ni,nj = i + col[k,0],j + col[k,1]
                        mat2[ni,nj] = max(2,mat2[ni,nj])

            else: # if the cell is ureceptive, accounf for diffusion
                levels2[i,j] -= alpha2 * levels1[i,j]

    return update_flag

def idx_to_coord(i,j,a=1):
    '''
    :param i: row index
    :param j: column index
    :param a: the hex side length
    :return: x and y coordinates
    '''
    return 1.5*j*a, -SQ3 * (i + (0.5 if j%2 != 0 else 0))


def plot_mat(mat:np.ndarray,ax:plt.axes,a=1,annotate=False,clr=None,plt_setup=False):
    '''plots the matrix and indicates the cell states
    :param mat: the matrix of flagx
    :param ax: the axes object to plot onto
    :param a: the side length of the hexgrid
    :param annotate: whether to plot cell indices
    :param clr: the color list to replace clrs
    :param plt_setup: whether to set plot limits and aspect ratio'''

    clrs = ['#ff8400','#b5b5b5','#00c8ff','#009e25'] # colors for the surface type with the corresponding index

    N,M = mat.shape
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            x,y = idx_to_coord(i,j,a)
            if annotate: ax.annotate(f'({i},{j})',(x,y),ha='center',va='center')

            hexagon = RegularPolygon((x,y), numVertices=6, radius=a, alpha=0.5,edgecolor='#9e9e9e',
                                     orientation=30/180*np.pi,facecolor=clrs[mat[i,j]] if clr==None else clr[mat[i,j]])
            ax.add_patch(hexagon)

    if plt_setup:
        xmin, ymin = idx_to_coord(0,0,a)
        xmax, ymax = idx_to_coord(N,M,a)

        ax.set_xlim((xmin-a,xmax+a))
        ax.set_ylim((ymax-a,ymin+a))
        ax.set_aspect('equal')
        ax.axis('off')


def plot_levels(mat_flags:np.ndarray,mat_levels:np.ndarray,ax:plt.axes,a=1,annotate=False,plt_setup=False):
    '''plots the matrix and indicates the cell levels
    :param mat_flags: the matrix of flagx
    :param mat_levels: the matrix of levels
    :param ax: the axes object to plot onto
    :param a: the side length of the hexgrid
    :param annotate: whether to display cell levels
    :param plt_setup: whether to impose plot limits, aspect ratio and turn of axes'''

    clrs = ['#ff8400','#b5b5b5','#00c8ff','#009e25']

    N,M = mat_flags.shape
    for i in range(mat_flags.shape[0]):
        for j in range(mat_flags.shape[1]):
            x,y = idx_to_coord(i,j,a)
            if annotate: ax.annotate(f'{mat_levels[i,j]*100:.0f}',(x,y),ha='center',va='center')

            hexagon = RegularPolygon((x,y), numVertices=6, radius=a, alpha=0.5,edgecolor=clrs[mat_flags[i,j]],
                                     orientation=30/180*np.pi,facecolor=clrs[mat_flags[i,j]])
            ax.add_patch(hexagon)

    if plt_setup:
        xmin, ymin = idx_to_coord(0,0,a)
        xmax, ymax = idx_to_coord(N,M,a)

        ax.set_xlim((xmin-a,xmax+a))
        ax.set_ylim((ymax-a,ymin+a))
        ax.set_aspect('equal')
        ax.axis('off')


class MatPlot():
    def __init__(self,mat:np.ndarray,a=1) -> None:

        self.clrs = ['#ff8400','#b5b5b5','#00c8ff','#009e25']
        self.hexes = [[] for _ in range(mat.shape[0])]

        self.shape = mat.shape
        self.mat = mat.copy()

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                x,y = idx_to_coord(i,j,a)
                self.hexes[i].append(RegularPolygon((x,y), numVertices=6, radius=a, alpha=0.5,orientation=30/180*np.pi,facecolor=self.clrs[mat[i,j]],edgecolor='#9e9e9e'))
    
    def update(self,mat:np.ndarray):
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if self.mat[i,j] == mat[i,j]: continue
                self.hexes[i][j].set_facecolor(self.clrs[mat[i,j]])

        self.mat = mat.copy()


    def add_to_plot(self,ax):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                ax.add_patch(self.hexes[i][j])

def hex_demo():
    N = 10
    M = 11

    mark_i = 5
    mark_j = 4
    mark = 2

    # declare the surface; 0 is for the edge, 1 is for unreceptive, 2 for boundary and 3 for frozen
    mat = np.zeros((N,M),dtype=int)

    mat[1:-1,1:-1] = 1 # set all the inner particles to unreceptive

    mat2 = mat.copy()

    mat[mark_i,mark_j] = 3


    for i in range(6):
        col = NEIGHS_EVEN_COL if mark_j%2 == 0 else NEIGHS_ODD_COL
        mat[mark_i+col[i,0],mark_j+col[i,1]] = mark

    clrs = ['#ff8400','#b5b5b5','#00c8ff','#009e25'] # colors for the surface type with the corresponding index

    coords = np.array([[row,col] for row in range(N) for col in range(M)])

    a = 1

    xy_coords = coords.copy().astype(float)

    xy_coords[:,1] *= 1.5*a
    xy_coords[:,0] *= -np.sqrt(3)*a
    xy_coords[coords[:,1]%2==1,0] -= np.sqrt(3)/2*a


    for _ in range(5):
        fig, ax = plt.subplots(1,2,figsize=(16,8))
        for i in range(N):
            for j in range(M):
                x,y = idx_to_coord(i,j,a)
                ax[0].annotate(f'({i},{j})',(x,y),ha='center',va='center')
                ax[1].annotate(f'({i},{j})',(x,y),ha='center',va='center')

                hexagon = RegularPolygon((x,y), numVertices=6, radius=a, alpha=0.5, edgecolor='k',orientation=30/180*np.pi,color=clrs[mat[i,j]])
                ax[0].add_patch(hexagon)

                hexagon = RegularPolygon((x,y), numVertices=6, radius=a, alpha=0.5, edgecolor='k',orientation=30/180*np.pi,color=clrs[mat2[i,j]])
                ax[1].add_patch(hexagon)

        for i in ax:
            i.set_xlim((xy_coords[:,1].min()-a,xy_coords[:,1].max()+a))
            i.set_ylim((xy_coords[:,0].min()-a,xy_coords[:,0].max()+a))
            i.set_aspect('equal')
            i.axis('off')

        plt.tight_layout()
        plt.show()
        plt.close()

        mat, mat2 = mat2, mat

def animate():
    '''to stack the frames use
    ffmpeg -framerate <framerate> -i %d.png <video_name>.<filetype_suffix>
    '''
    beta = 0.9
    gamma = 0.001
    alpha = 1
    a = 1

    M = 55
    N = 51

    # setup the flag matrices
    mat1 = np.zeros((N,M),dtype=int)
    mat1[1:-1,1:-1] = 1
    outside_mask = mat1 == 0
    mat1[int(N/2),int(M/2)] = 3

    # set the neighbor cells to the frozen one to receptive
    col = NEIGHS_EVEN_COL if int(M/2)%2 == 0 else NEIGHS_ODD_COL
    for i in range(6):
        ni,nj = int(N/2) + col[i,0], int(M/2) + col[i,1]
        mat1[ni,nj] = 2

    mat2 = mat1.copy()

    # setup the water level matrices
    levels1 = np.ones((N,M),dtype=float) * beta
    levels2 = levels1.copy()
    # no adjusting of the water level of the initial frozen cell since the water level in frozen/neighbor cells is irrelevant

    plot_hex = MatPlot(mat1)

    #plt.ion() # uncomment if you want to watch the animation in rela time

    fig, ax = plt.subplots(1,figsize=(10,10))
    plot_hex.add_to_plot(ax)

    ttl = ax.set_title('Step 0')
    xmin, ymin = idx_to_coord(0,0,a)
    xmax, ymax = idx_to_coord(N,M,a)

    ax.set_xlim((xmin-a,xmax+a))
    ax.set_ylim((ymax-a,ymin+a))
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    # plt.show()

    cnt = 0
    for i in range(30):
        change = simulate_step(levels1,levels2,mat1,mat2,alpha,gamma)

        levels1 = levels2.copy()
        mat1 = mat2.copy()
        print(i)
        ttl.set_text(f'Step {i+1}')
        if change:
            plot_hex.update(mat1)
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            #plot_mat(mat1,ax[0],a)

            #plt.close()
            plt.savefig(f'frames/{cnt}.png') # save the figure
            cnt += 1
            time.sleep(0.1)
        
        if 2 in mat1[outside_mask]: break

def animate2():
    '''to stack the frames use
    ffmpeg -framerate <framerate> -i %d.png <video_name>.<filetype_suffix>
    '''
    beta = 0.9
    gamma = 0.001
    alpha = 1
    a = 1

    M = 13
    N = 13

    # setup the flag matrices
    mat1 = np.zeros((N,M),dtype=int)
    mat1[1:-1,1:-1] = 1
    outside_mask = mat1 == 0
    mat1[int(N/2),int(M/2)] = 3

    # set the neighbor cells to the frozen one to receptive
    col = NEIGHS_EVEN_COL if int(M/2)%2 == 0 else NEIGHS_ODD_COL
    for i in range(6):
        ni,nj = int(N/2) + col[i,0], int(M/2) + col[i,1]
        mat1[ni,nj] = 2

    mat2 = mat1.copy()

    # setup the water level matrices
    levels1 = np.ones((N,M),dtype=float) * beta
    levels2 = levels1.copy()
    # no adjusting of the water level of the initial frozen cell since the water level in frozen/neighbor cells is irrelevant


    cnt = 0
    for i in range(20):
        change = simulate_step(levels1,levels2,mat1,mat2,alpha,gamma)

        levels1 = levels2.copy()
        mat1 = mat2.copy()
        print(i)
        if change:
            fig, ax = plt.subplots(1,figsize=(10,10))

            plot_levels(mat1,levels1,ax,annotate=True,plt_setup=True)
            plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
            plt.show()
            plt.close()
        
        if 2 in mat1[outside_mask]: break

def border_analysis():
    N = 20
    M = 31
    a = 1

    bw_i = 0 # border width in the y direction
    bw_j = 1 # border widht in the x direction
    mark_i = 5
    mark_j = 3
    len_i = 5
    len_j = 9

    clrs = ['#b5b5b5','r','g','b','c','m','y','orange']
    # declare the surface; 0 is for the edge, 1 is for unreceptive, 2 for boundary and 3 for frozen
    mat = np.zeros((N,M),dtype=int)

    mat[mark_i:mark_i + len_i,mark_j:mark_j + len_j] = 6 # mark the whole area
    mat[len_i + mark_i,mark_j:mark_j + len_j] = 5 # next bottom row
    mat[mark_i:mark_i + len_i,mark_j + len_j] = 4 # next right
    mat[mark_i-1,mark_j:mark_j + len_j] = 3 # next top row
    mat[mark_i:mark_i + len_i,mark_j-1] = 2
    if (mark_j + len_j)%2 == 0:
        mat[mark_i + len_i,mark_j + len_j ] = 1
    else:
        mat[mark_i - 1,mark_j + len_j] = 1

    if mark_j%2 == 0:
        mat[mark_i - 1,mark_j - 1] = 1
    else:
        mat[mark_i+ len_i,mark_j - 1] = 1

    mat[mark_i:mark_i + len_i,mark_j+len_j:mark_j + 2*len_j] = 7
    fig,ax = plt.subplots(1,figsize=(15,10))

    plot_mat(mat,ax,a,clr=clrs)

    xmin, ymin = idx_to_coord(0,0,a)
    xmax, ymax = idx_to_coord(N,M,a)

    ax.set_xlim((xmin-a,xmax+a))
    ax.set_ylim((ymax-a,ymin+a))
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()

    plt.show()
    plt.close()


def mark_neighs(x0,x1,y0,y1,bw:int,c,ax:plt.axes):
    '''marks neighbors
    x0,x1 are index limits in the interval [x0,x1) and the same for y'''
    #mat[x1,y0:y1] = 5 # bottom edge
    boundary_elements = []
    for i in range(y0,y1):
        for j in range(bw):
            x,y = idx_to_coord(x1+j,i,1)
            boundary_elements.append([x,y])
    boundary_elements = np.array(boundary_elements)
    ax.scatter(boundary_elements[:,0],boundary_elements[:,1],edgecolors=c,marker='.',s=400,facecolors='none')

    #mat[x0:x1,y1] = 4 # right edge
    boundary_elements = []
    for i in range(x0,x1):
        for j in range(bw):
            x,y = idx_to_coord(i,y1+j,1)
            boundary_elements.append([x,y])
    
    boundary_elements = np.array(boundary_elements)
    ax.scatter(boundary_elements[:,0],boundary_elements[:,1],c=c,marker='+',s=200)

    #mat[x0-1,y0:y1] = 3 # top edge
    boundary_elements = []
    for i in range(y0,y1):
        for j in range(bw):
            x,y = idx_to_coord(x0-1-j,i,1)
            boundary_elements.append([x,y])
    boundary_elements = np.array(boundary_elements)
    ax.scatter(boundary_elements[:,0],boundary_elements[:,1],edgecolors=c,marker='.',s=400,facecolors='none')

    #mat[x0:x1,y0-1] = 2 # left edge
    boundary_elements = []
    for i in range(x0,x1):
        for j in range(bw):
            x,y = idx_to_coord(i,y0-1-j,1)
            boundary_elements.append([x,y])
    
    boundary_elements = np.array(boundary_elements)
    ax.scatter(boundary_elements[:,0],boundary_elements[:,1],c=c,marker='+',s=200)
    
    # handle the top/bottom left/right neighbors

    if y1%2 == 0: # if the right edge column index is odd the bottom right neighbor is touching the main square (y1 is the limit + 1)
        boundary_elements = []
        for i in range(bw):
            for j in range(bw):
                x,y = idx_to_coord(x1+i,y1+j,1) # handle the bottom right neighbor
                boundary_elements.append([x,y])
                x,y = idx_to_coord(x0-1-i,y1+j,1) # handle the upper right neighbor
                boundary_elements.append([x,y])
        
        boundary_elements = np.array(boundary_elements)
        ax.scatter(boundary_elements[:,0],boundary_elements[:,1],c=c,marker='x',s=100)
    else:
        boundary_elements = []
        for i in range(bw):
            for j in range(bw):
                x,y = idx_to_coord(x0-i-1,y1+j,1) # handle the top right neighbor
                boundary_elements.append([x,y])
                x,y = idx_to_coord(x1+i,y1+j,1) # handle the lower right neighbor
                boundary_elements.append([x,y])
        
        boundary_elements = np.array(boundary_elements)
        ax.scatter(boundary_elements[:,0],boundary_elements[:,1],c=c,marker='x',s=100)


    if y0%2 == 0: # in case the left index is even the top left neighbor touches the square, otherwise it is the bottom left
        boundary_elements = []
        for i in range(bw):
            for j in range(bw):
                x,y = idx_to_coord(x0-1-i,y0-1-j,1) # handle the bottom right neighbor
                boundary_elements.append([x,y])
                x,y = idx_to_coord(x1+i,y0-1-j,1) # handle the upper right neighbor
                boundary_elements.append([x,y])
        
        boundary_elements = np.array(boundary_elements)
        ax.scatter(boundary_elements[:,0],boundary_elements[:,1],c=c,marker='x',s=100)
    else:
        boundary_elements = []
        for i in range(bw):
            for j in range(bw):
                x,y = idx_to_coord(x1+i,y0-1-j,1) # handle the bottom left neighbor
                boundary_elements.append([x,y])
                x,y = idx_to_coord(x0-i-1,y0-1-j,1) # handle the upper left neighbor
                boundary_elements.append([x,y])
        
        boundary_elements = np.array(boundary_elements)
        ax.scatter(boundary_elements[:,0],boundary_elements[:,1],c=c,marker='x',s=100)

    
def tile_the_plane():
    N = 51
    M = 51
    a = 1

    N_x = 5
    N_y = 1

    bw = 2 # border width
    edge = 1

    clrs = ['#ff8400','r','g','b','c','m','y','orange']
    #clrs = ['#808080' for _ in clrs]
    clrs[5] = '#8fffa0'
    # declare the surface; 0 is for the edge, 1 is for unreceptive, 2 for boundary and 3 for frozen
    mat = np.zeros((N,M),dtype=int)

    fig,ax = plt.subplots(1,figsize=(15,15))

    for i in range(N_x):
        for j in range(N_y):
            x0, x1 = int(i*(N-2*edge)/N_x) + edge, int((i+1)*(N-2*edge)/N_x) + edge
            y0, y1 = int(j*(M-2*edge)/N_y) + edge, int((j+1)*(M-2*edge)/N_y) + edge
            mat[x0:x1,y0:y1] = (N_y*i + j)%6 + 1    

    plot_mat(mat,ax,a,clr=clrs)

    for i in range(N_x):
        for j in range(N_y):
            x0, x1 = int(i*(N-2*edge)/N_x) + edge, int((i+1)*(N-2*edge)/N_x) + edge
            y0, y1 = int(j*(M-2*edge)/N_y) + edge, int((j+1)*(M-2*edge)/N_y) + edge
            
            mark_neighs(x0,x1,y0,y1,bw,clrs[mat[x0,y0]],ax)

    xmin, ymin = idx_to_coord(0,0,a)
    xmax, ymax = idx_to_coord(N,M,a)

    #ax.set_xlim((xmin-a,xmax+a))
    #ax.set_ylim((ymax-a,ymin+a))
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()

    plt.show()
    plt.close()


def strip_the_plane():

    N = 3
    n = 13
    m = 13
    b = 2

    mat = load_from_bin('data/test_sflk_0.bin',13,13)#np.zeros((n,m),dtype=int)#
    #mat[1:-1,1:-1] = 1
    counts = [int((i+1)*(n-2)/N) - int(i*(n-2)/N) + 2*b for i in range(N)]
    counts[0] = int((n-2)/N) + 1 + b 
    counts[-1] = n - int((N-1)*(n-2)/N) - 1 + b
    offsets = [0,*[int((i+1)*(n-2)/N) - b + 1 for i in range(N-1)]]
    return_counts = [int((i+1)*(n-2)/N) - int(i*(n-2)/N) for i in range(N)]
    print("return_counts: ",return_counts)

    print(counts)
    print(offsets)

    fig,ax = plt.subplots(1,figsize=(10,10))

    #clrs = ['#ff8400','r','g','b','c','m','y','orange']
    clrs = ['#ff8400','#b5b5b5','#00c8ff','#009e25','r','y','b','m']

    for j,offset in enumerate(offsets):
        mask = mat[1+int(j*(n-2)/N):1+int((j+1)*(n-2)/N),1:-1] == 1
        mat[1+int(j*(n-2)/N):1+int((j+1)*(n-2)/N),1:-1][mask] = 4+j%5 
    #mat[int(n/2),int(m/2)] = 0
    plot_mat(mat,ax,clr=clrs)#

    for j,offset in enumerate(offsets):
        coords = []
        for i in range(m):
            x,y = idx_to_coord(offset,i) # top limit including buffer
            coords.append([x,y])
            x,y = idx_to_coord(offset+counts[j]-1,i) # bottom limit including buffer
            coords.append([x,y])
            x,y = idx_to_coord(int(j*(n-2)/N)+1,i) # top limit without buffer
            coords.append([x,y])

        coords = np.array(coords)

        ax.scatter(coords[:,0],coords[:,1],c=clrs[4+j%5])

    plt.axis('equal')
    plt.show()
    plt.close()


def analyze_intermediate():
    nsteps = 7
    ranks = 3
    ns = [19,20,20]
    fig,ax = plt.subplots(nsteps,2*ranks)

    for i in range(nsteps):
        for j in range(ranks):
            arr_pre = load_from_bin(f'data/rank_{j}_step_{2*i}_pre.bin',ns[j],55)
            arr_post = load_from_bin(f'data/rank_{j}_step_{2*i}_post.bin',ns[j],55)

            plot_mat(arr_pre,ax[i,2*j],plt_setup=True)
            plot_mat(arr_post,ax[i,2*j + 1],plt_setup=True)


    plt.show()
    plt.close()


if __name__ == "__main__":
    #analyze_intermediate()
    strip_the_plane()
    #animate2()

    #animate()
    #tile_the_plane()