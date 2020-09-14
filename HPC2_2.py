from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg
from scipy.linalg import norm
import matplotlib.pyplot as plt
from pylab import imshow, jet
import time

class Timer:
    '''
    Class for measuring time, returns time interval
    taken by the code execution.
    '''
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


#function implementing OpenCL five point stencil
def mv(u):
    '''
    Takes a matrix U in a flattened vector form u, and applies the
    five point stencil of the Laplace operator in 2D to the interior points.
    Returns the result in a matrix format also flattened to a vector.
    LinearOperator with this funcion for _matvec() can be used to solve the 
    Laplace/Poisson equation with Dirichlet boundary conditions using the 
    Scipy CG iterative solver.
    '''
    M = np.int32(np.sqrt(len(u)))

    platform = cl.get_platforms()[0]    # Select the first platform [0]
    device = platform.get_devices()[0]  # Select the first device on this platform [0]
    ctx = cl.Context([device])
    # Create (non-custom) command queue
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    mf = cl.mem_flags             # Memory flags
        
    #Create a buffer on the device containing the vector u
    uDevice = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=u)

    resDevice = cl.Buffer(ctx, mf.WRITE_ONLY, u.nbytes)

    stencil_kernel = """
    __kernel void fpStencil(const __global double* u,
                            const int uSize,
                            const int M,
                            __global double* resU) 
    {

        int gid = get_global_id(0);

        resU[gid] = (4 * u[gid] - u[gid - M] - u[gid + M] - u[gid - 1] - u[gid + 1]) * M * M;

        if (gid < M) {
            resU[gid] = 0;
        }

        if  (gid > uSize - M) {
            resU[gid] = 0;
        }

        if (gid%M == 0) {
            resU[gid] = 0;
        }

        if (gid%M == M - 1) {
            resU[gid] = 0;
        }
    }
    """

    resU = np.empty_like(u)
    prg = cl.Program(ctx, stencil_kernel).build()
    prg.fpStencil(queue, u.shape, None, uDevice, np.int32(u.size), M, resDevice).wait()
    cl.enqueue_copy(queue, resU, resDevice).wait()

    return np.float64(resU)

def runCG(N):
    '''
    creates a LinearOperator corresponding to the problem on an N*N grid,
    and the appropriate RHS vectr and uses the linalg.cg function to solve
    the system iteratively. It also uses the callback function to store 
    the residual at each step, and the total number of iterations.
    '''

    #RHS matrix including boundary conditions, then flattened to vector
    fMat = np.ones((N,N))
    for i in range(N):
        for j in range(N):
            if(i == 0 or j == 0 or i == N-1 or j == N-1):
                fMat[i][j] = 0
    fVec = fMat.flatten().astype('float64')

    #Random starting matrix U, flattened to vector
    uMat = np.random.rand(N,N)
    for i in range(N):
        for j in range(N):
            if (i == 0 or j == 0 or i == N-1 or j == N-1):
                uMat[i][j] = 0

    uVec = uMat.flatten()


    def cgCallback(xk):
        '''
        Function to extract the residual at each iteration, which allows us to
        study the convergence of the solution.
        '''
        r = norm(fVec - mv(xk)) / norm(fVec)
        errors.append(r)

    errors = []
    boo = LinearOperator(shape = (N*N,N*N), matvec = mv, dtype = np.float64)
    with Timer() as t1:
        resVec = cg(boo, np.float64(fVec), callback = cgCallback)
    print(t1.interval)
    resMat = np.asarray(np.reshape(resVec[0], (N,N)))
    iterations = len(errors)

    output = [resMat, errors, iterations]
    
    return output

### generate runs for 5 <= N <= 200
iter = []
err = []
results = []

### Takes some time to run, prints progress every 5 passes
for n in np.arange(5,200,10):
    run = runCG(n)
    iter.append(run[2])
    err.append(run[1])
    results.append(run[0])



'''
plt.plot(iter)

%matplotlib inline

for i in range(len(err)):
    plt.plot(np.arange(1,len(err[i])+1), err[i], label = "M="+str(i*10+5))
    
plt.legend(loc='upper right')

asd = results[90]
imshow(asd)
'''
