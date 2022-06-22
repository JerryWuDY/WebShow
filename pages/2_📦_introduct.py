import streamlit as st
# from PIL import Image

from CommonConfig import barPy
barPy.Websidebar()

st.markdown('# Cuda Introduction \n'
            '## 什么是CUDA？')
st.markdown('* **CUDA**\n' 
            'Compute Unified Device Architecture \n'
            '* **Support** C/C++/Python \n'
            '基于C/C++的编程方法。支持异构编程的扩展方法，简单明了的APIs，能够轻松管理储存系统')

with st.expander("Click here to see the Difference!"):
    img1 = "https://github.com/JerryWuDY/WebShow/raw/main/src/CUDA/GPUvsCPU.png"
    st.image(img1)


st.markdown("## 适用设备:\n"
    "- 所有包含NVIDIA GPU的服务器，工作站，个人电脑，嵌入式设备等电子设备"
    "- 软件安装: \n"
    "Windows：https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html \n"
    "只需安装一个.exe的可执行程序"
            )

code_NVCC = '''!nvcc -V # 可以检查你是否安装好CUDA
!nvidia-smi # 查看是否有支持CUDA的NVIDIA GPU'''
st.code(code_NVCC,language="python")


st.markdown(
    "## GPU的硬件结构 \n"
    "下图所示的是GA100的硬件架构图，它包含了：\n"
    "- 8192 FP32 CUDA Cores（用于计算的核心）\n"
    "- 128个SM（SM指stream multiprocessor，即流多处理器，可以方便一块线程之间的协作）\n"
    "- 每个SM包含64个FP32 CUDA Core，4个第三代Tensor Core"
)

with st.expander("- Device"):
    imgHardware = "https://github.com/JerryWuDY/WebShow/raw/main/src/CUDA/hardware.png"
    st.image(imgHardware)

with st.expander("- SM"):
    imgSM = "https://github.com/JerryWuDY/WebShow/raw/main/src/CUDA/sm.png"
    st.image(imgSM)


st.markdown(
    "## CUDA的线程层次 \n"
    "在计算机科学中，执行线程是可由调度程序独立管理的最小程序指令序列。\n"
    "在GPU中，可以从多个层次管理线程：\n"
    "- Thread: sequential execution unit 所有线程执行相同的核函数  并行执行\n"
    "- Thread Block: a group of threads  执行在一个Streaming Multiprocessor (SM)  同一个Block中的线程可以协作\n"
    "- Thread Grid: a collection of thread blocks  一个Grid当中的Block可以在多个SM中执行"
)
imgThread = "https://github.com/JerryWuDY/WebShow/raw/main/src/CUDA/thread.png"
st.image(imgThread)

st.markdown(
    "## CUDA程序的编写 \n"
    "*kernel* 函数的实现\n\t需要在核函数之前加上 **@cuda.jit**标识符  \n"
)
st.code('''@cuda.jit 
    def add_kernel(x,y,out)''',language="python")
st.markdown(
    "*kernel* 函数的调用  \n"
    "需要添加执行设置 \n"
    "**add_kernel[blocks_per_grid, threads_per_block](x, y, out)**  \n"
    "这里的blocks_per_grid代表Grid中block在x,y,z三个维度的数量  \n"
    "这里的threads_per_block代表Block中thread在x,y,z三个维度的数量 "
)


st.markdown(
    "## CUDA 线程索引"
)
imgthreadIndex = "https://github.com/JerryWuDY/WebShow/raw/main/src/CUDA/thread_index2_python.png"
imgPixel = "https://github.com/JerryWuDY/WebShow/raw/main/src/CUDA/Pixel.jpg"
st.image(imgthreadIndex)
st.image(imgPixel)


st.markdown(
    "## 矩阵相乘 \n"
    "矩阵操作在很多领域都有非常广泛的应用，比如在非常热门的卷积神经网络中的卷积操作，就可以利用矩阵乘来完成。接下来，我们就尝试利用CUDA来加速矩阵相乘的操作。"
    "下面展示了如何利用CPU来完成矩阵相乘的操作"
)
code_matmul_CPU ='''
def matmul_cpu(A,B,C):
	for y in range(B.shape[1]):
		for x in range(A.shape[0]):
			tmp = 0
			for k in range(A.shape[1]):
				tmp += A[x,k]*B[k,y]
			C[x,y] = tmp
'''
st.code(code_matmul_CPU,language="python")

st.markdown("实现CUDA核函数")
code_matmul_GPU_kernel = '''
@cuda.jit
def matmul_gpu(A,B,C):
    row,col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row,k]*B[k,col]
        C[row,col] = tmp
'''
st.code(code_matmul_GPU_kernel)

st.markdown("第三步，利用SM中的Shared memory来优化核函数")
code_matmul_GPU_share='''
TPB = 16
@cuda.jit
def matmul_shared_mem(A,B,C):
    sA = cuda.shared.array(shape=(TPB,TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB,TPB), dtype=float32)

    x,y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    if x>=C.shape[0] or y >= C.shape[1]:
        return
    tmp = 0.
    for i in range(int(A.shape[1]/TPB)):
        sA[tx, ty] = A[x, ty+i*TPB]
        sB[tx, ty] = B[tx+i*TPB, y]
        cuda.syncthreads()
        for j in range(TPB):
            tmp += sA[tx,j]*sB[j,ty]
    C[x,y] = tmp
'''
st.code(code_matmul_GPU_share)

st.markdown("第四步，定义main函数，在这部中，我们初始化A，B矩阵，并将数据传输给GPU")
code_matmul_GPU_main='''
def main_matrix_mul():
    TPB = 16
    cal_times = 50 #构建矩阵
    A = np.full((TPB*cal_times,TPB*cal_times), 3.0, float)
    B = np.full((TPB*cal_times,TPB*cal_times), 4.0, float)
    C_cpu = np.full((A.shape[0],B.shape[1]), 0, np.float)
    
    #Start in CPU
    print("Start processing in CPU")
    start_cpu = time.time()
    matmul_cpu(A,B,C_cpu)
    end_cpu = time.time()
    time_cpu = (end_cpu - start_cpu)
    print("CPU time: "+str(time_cpu))
    
    #Start in GPU
    A_global_mem = cuda.to_device(A)
    B_global_mem = cuda.to_device(B)
    
    C_global_mem = cuda.device_array((A.shape[0],B.shape[1]))
    C_shared_mem = cuda.device_array((A.shape[0],B.shape[1]))
    
    threadsperblock = (TPB, TPB)
    blockspergrid_x = int(math.ceil(A.shape[0]/threadsperblock[0]))
    blockspergrid_y = int(math.ceil(A.shape[1]/threadsperblock[1]))
    blockspergrid = (blockspergrid_x,blockspergrid_y)
    
    print("Start processing in GPU")
    start_gpu = time.time()
    matmul_gpu[blockspergrid, threadsperblock](A_global_mem,B_global_mem,C_global_mem)
    cuda.synchronize()
    end_gpu = time.time()
    time_gpu = (end_gpu - start_gpu)
    print("GPU time(global memory):"+str(time_gpu))
    C_global_gpu = C_global_mem.copy_to_host()
    
    print("Start processing in GPU (shared memory)")
    start_gpu = time.time()
    matmul_shared_mem[blockspergrid, threadsperblock](A_global_mem,B_global_mem,C_global_mem)
    cuda.synchronize()
    end_gpu = time.time()
    time_gpu = (end_gpu - start_gpu)
    print("GPU time(shared memory):"+str(time_gpu))
    C_shared_gpu = C_shared_mem.copy_to_host
'''
st.code(code_matmul_GPU_main)
st.markdown("执行main函数，进行对比（采用实验室电脑）")

imgResult = "https://github.com/JerryWuDY/WebShow/raw/main/src/CUDA/result_compare.png"
st.image(imgResult)
st.write("倍数：",88.11427187919617/0.0820000171661377)
st.markdown("If you want to know more, you can search for **{0}** or **{1}**  \n \t which you have to know the computer knowledge".format("Cython","Numba"))