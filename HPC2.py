from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csr_matrix
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


def rand_csr(N,density=0.4):
    '''
    Generate a random sparce matrix in CSR format of the specified density.
    '''
    
    # Create random square sparse matrix in COO format
    coo_sparse = sparse.random(N,N,density,dtype='float32')
    # Convert to CSR format and return
    return sparse.csr_matrix(coo_sparse)     # Convert to CSR format and return


class Sparce(LinearOperator):
    '''
    Class derived from LinearOperator, initialised with a sparce matrix in
    the CSR format. It overrides the _matvec function to implement 
    algorithm for CSR multiplication, accelerated by OpenCL.
    '''

    def __init__(self, inMat):
        '''
        Initiaises Sparce matrix with input type scipy.sparse.csr_matrix.
        shape and dtype fields are required by LinearOperator. 
        '''
        self.shape = inMat.get_shape()
        self.dtype = 'float32'
        self.data = inMat.data
        self.indices = inMat.indices
        self.indptr = inMat.indptr
        
    def _matvec(self, v):
        '''
        OpenCL implementation of product of sparce matrix object with vector v.
        Returns A*v. Dimensions of v should match first dimension of the sparce matrix
        '''
        platform = cl.get_platforms()[0]    # Select the first platform [0]
        device = platform.get_devices()[0]  # Select the first device on this platform [0]
        ctx = cl.Context([device])
        queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)  # Create (non-custom) command queue
        mf = cl.mem_flags             # Memory flags
        
        # 4 compute units available on my CPU (2 cores + HT)
        # AVX2 -> 256 bit AVX --> double4
  
        vec =  v.astype(np.float32)
        data = self.data.astype(np.float32)    # Data vector
        ind = self.indices.astype(np.int32)   # Vector of column indices
        indptr  = self.indptr.astype(np.int32)    # Index pointer of column

        #Create buffers
        data_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
        ind_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ind)
        indptr_buff  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=indptr)
        vec_buff  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vec)
        res_buff = cl.Buffer(ctx,mf.WRITE_ONLY,vec.nbytes)

        N = np.float32(len(indptr) - 1)    # Length of the output vector

        ### Create a kernel which carries out the CSR multiplication
        matvec_kernel ="""
        __kernel void matvec( const int N,
                          __global const float * data,
                          __global const int * index,
                          __global const int * indptr,
                          __global const float * vec,
                          __global float * res )
        {
        
        int gid = get_global_id(0);
        
        float acc = 0.0f;
        
        for (int k = indptr[gid]; k < indptr[gid + 1]; k++)
            {
            acc += data[k] * vec[index[k]];
            }
        
        res[gid] = acc;
        } """


        prg = cl.Program(ctx,matvec_kernel).build()

        res_np = np.empty_like(vec).astype(np.float32)
        prg.matvec(queue,vec.shape,None,np.int32(N),
            data_buff,ind_buff,indptr_buff,vec_buff,res_buff).wait()
        cl.enqueue_copy(queue, res_np, res_buff).wait()

        return res_np


    
    def _matvecsimd(self, v):
        '''
        Vectorised sparse matrix product.
        Breaks problem down into double4 vector types, carries out multiplications
        in vector form, then accumulates the values corresponding to each matrix
        row, therefore to each result element.
        '''
        platform = cl.get_platforms()[0]    # Select the first platform [0]
        device = platform.get_devices()[0]  # Select the first device on this platform [0]
        ctx = cl.Context([device])
        queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)  # Create (non-custom) command queue
        mf = cl.mem_flags             # Memory flags
        
        # 4 compute units available on my CPU (2 cores + HT)
        # AVX2 -> 256 bit AVX --> double4

        vec =  v.astype(np.float32)
        data = self.data.astype(np.float32)    # Data vector
        ind = self.indices.astype(np.int32)   # Vector of column indices
        indptr  = self.indptr.astype(np.int32)    # Index pointer of column

        remainder = np.uint32(np.size(data)%4)

        ### Pad data and indices to be divisible by 8
        if remainder != 0:
            data = np.concatenate([data, np.zeros(4 - remainder)]).astype(np.float32)
            ind = np.concatenate([ind, np.zeros(4 - remainder)]).astype(np.uint32)

        iteration_len = np.size(data)

        #Create buffers
        data_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
        ind_buff = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ind)
        indptr_buff  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=indptr)
        vec_buff  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=vec)
        res_buff = cl.Buffer(ctx,mf.WRITE_ONLY,vec.nbytes)
        res_tmp_buff = cl.Buffer(ctx, mf.READ_WRITE, data.nbytes)


        matvec_simd_kernel ="""
        __kernel void mult(__global const float * data,
                            __global const int * index,
                            __global const float * vec,
                            __global float * resTmp )
        {
        
            int gid = get_global_id(0);
            
            double4 dataSlice;
            double4 vecSlice;
            double4 resSlice;

            dataSlice = (double4)(data[gid*4+0],
                                data[gid*4+1],
                                data[gid*4+2],
                                data[gid*4+3]);

            vecSlice = (double4)(vec[index[gid*4+0]],
                                vec[index[gid*4+1]],
                                vec[index[gid*4+2]],
                                vec[index[gid*4+3]]);

            resSlice = dataSlice * vecSlice;

            resTmp[gid*4+0] = resSlice.s0;
            resTmp[gid*4+1] = resSlice.s1;
            resTmp[gid*4+2] = resSlice.s2;
            resTmp[gid*4+3] = resSlice.s3;
        
        }

        __kernel void add(__global const int* indptr,
                                __global const float* resTmp,
                                __global float* res) {

            int gid = get_global_id(0);
            float acc = 0.0f;

            for(uint j = indptr[gid]; j < indptr[gid+1]; j++){
                acc += resTmp[j];
            }

            res[gid] = acc;
        }
        """


        prg = cl.Program(ctx,matvec_simd_kernel).build()

        prg.mult(queue, (int(iteration_len/4), ), None,
                    data_buff, ind_buff, vec_buff, res_tmp_buff)

        prg.add(queue, (np.size(indptr) - 1,), None, 
                    indptr_buff, res_tmp_buff, res_buff)

        res = np.empty_like(v)
        cl.enqueue_copy(queue, res, res_buff).wait()

        # Trim result so only the original size vector is returned without padding.
        return res[0:np.size(v)]

def runCode(N):
    '''
    Generates a random Sparce matrix of dimensions N*N in CSR format.
    Also generates a random vector of length N, and carries out the 
    multiplication using both OpenCL functions, with and without SIMD optimisation.
    '''
    vec = np.random.rand(N).astype(np.float32)
    foo = rand_csr(N, 0.10)
    bar = Sparce(foo)

    with Timer() as t1:
        a = bar._matvec(vec)

    with Timer() as t2:
        b = bar._matvecsimd(vec)

    print("time to run without SIMD: {0}".format(t1.interval))
    print("time to run with SIMD: {0}".format(t2.interval))

runCode(10000)