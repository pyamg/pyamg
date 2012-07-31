import numpy
from array import array

__all__ = ['cvoutput2d', 'outputML']

# ConnectionViewer output package
# download ConnectionViewer from
# http://gcsc.uni-frankfurt.de/Members/mrupp/connectionviewer/

def cvoutput2d(filename, V, A):
    '''
    Helper function for outputML:
    Filename as string, V=array of vertices, A=sparse matrix
    '''
    cvoutput2dv(filename, numpy.asarray(V[:,0]), numpy.asarray(V[:,1]), A)

def cvoutput2dv(filename, Vx, Vy, A):
    '''
    Helper function for outputML:
    Filename as string, Vx=array of x-coordinates, Vy=array of y-coordinates, A=sparse matrix
    '''
    cvoutput2dp(filename, Vx, Vy, array('I', xrange(Vx.size)), array('I', xrange(Vy.size)), A)

def cvoutput2dp(filename, Vx, Vy, p1, p2, A):
    '''
    Helper function for outputML:
    Filename as string, Vx=array of x-coordinates, Vy=array of y-coordinates, 
    p1=map from row indices to Vx/Vy arrays
    p2=map from col indices to Vx/Vy arrays
    '''

    f = open(filename, 'w')
    f.write("1\n") # version
    f.write("2\n") # dimension
    f.write(repr(Vx.size) + "\n") # size
    for i in xrange(Vx.size):
        f.write(repr(Vx[i]) + " " + repr(Vy[i]) + "\n")
    f.write("1\n") # draw strings in window
    
    #for i in xrange(vertices.size/2):
    #   f.write(repr(i) + " " + repr(i) + " test" + "\n")
    A = A.tocoo()
    for i,j,v in zip(A.row, A.col, A.data):
        f.write(repr(int(p1[i])) + " " + repr(int(p2[j])) + " " + repr(v) + "\n")

def createMarks(marksFilename, mark, r, g, b, alpha):
    '''
    Helper function for outputML:
    Create a .mark file from a 0/1 mark array
    '''
    f = open(marksFilename, 'w')
    # r g b alpha size
    f.write(repr(r) + " " + repr(g) + " " + repr(b) + " " + repr(alpha) + " 0\n")
    for i in xrange(len(mark)):
        if mark[i] == 1:
            f.write(repr(i) + "\n")

def outputML(filename, V, mls):
    '''

    Parameters
    ----------
    filename : {string}
        filname root for ConnectionViewer data files where the data files are
        filename + '_' + ['A', 'P', 'R'] + level_number + '.mat'
        That is, a file is written for each of the three matrices, (A, P and R)
        on each level
    V : {array}
        (n x 2) array of vertices
    mls : {multilevel solver object}
        multilevel solver to visualize, with (n x n) finest level matrix
    
    Returns
    ------
    Files are written for each of the three matrices, (A, P and R) on each level
    in mls for ConnectionViewer to read.  The files allow for easy matrix
    stencil visualization.
    
    '''

    Vx = numpy.asarray(V[:,0])
    Vy = numpy.asarray(V[:,1])
    cvoutput2d(filename+"_A0.mat", V, mls.levels[0].A)
    for L in xrange(len(mls.levels)-1):
        R = mls.levels[L].R.tocoo()
        parent1=array('I', xrange(R.shape[1]))
        parent2=array('I', xrange(R.shape[0]))
        bCoarse=[0 for i in xrange(R.shape[1])]
        for i,j,v in zip(R.row, R.col, R.data):
            if v == 1.0:
                parent2[i] = j
                bCoarse[j] = 1
        bFine=[1-bCoarse[i] for i in xrange(R.shape[1])]

        cvoutput2dp(filename+"_P"+repr(L)+".mat", Vx, Vy, parent1, parent2, mls.levels[L].P)
        cvoutput2dp(filename+"_R"+repr(L)+".mat", Vx, Vy, parent2, parent1, mls.levels[L].R)

        createMarks(filename+"_coarse"+repr(L)+".marks", bCoarse, 0, 0, 1, 1)
        createMarks(filename+"_fine"+repr(L)+".marks", bFine, 1, 0, 0, 1)
        open(filename+"_A"+repr(L)+".mat", 'a').write("c "+filename+"_coarse"+repr(L)+".marks\nc "+filename+"_fine"+repr(L)+".marks\n")
        open(filename+"_P"+repr(L)+".mat", 'a').write("c "+filename+"_coarse"+repr(L)+".marks\nc "+filename+"_fine"+repr(L)+".marks\n")
        open(filename+"_R"+repr(L)+".mat", 'a').write("c "+filename+"_coarse"+repr(L)+".marks\nc "+filename+"_fine"+repr(L)+".marks\n")

        Vx = mls.levels[L].R * Vx
        Vy = mls.levels[L].R * Vy
        cvoutput2dp(filename+"_A"+repr(L+1)+".mat", Vx, Vy, parent1, parent1, mls.levels[L+1].A)
    #i = len(mls.levels)-1
    
