
__global__ void skepu_skel_0_SkePURenderers_MapReduceKernel_SkePUExposureFuncCalc_add_float(float* skepu_output, skepu::PRNG::Placeholder,class vec3 *col, float div,  size_t skepu_w2, size_t skepu_w3, size_t skepu_w4, size_t skepu_n, size_t skepu_base, skepu::StrideList<1> skepu_strides)
{
	extern __shared__ float sdata_skepu_skel_0[];
	
	size_t skepu_global_prng_id = blockIdx.x * blockDim.x + threadIdx.x;
	size_t skepu_blockSize = blockDim.x;
	size_t skepu_tid = threadIdx.x;
	size_t skepu_i = blockIdx.x * skepu_blockSize + skepu_tid;
	size_t skepu_gridSize = skepu_blockSize * gridDim.x;
	float skepu_result;
	if (skepu_strides[0] < 0) { col += (-skepu_n + 1) * skepu_strides[0]; }


	if (skepu_i < skepu_n)
	{
		
		skepu_result = skepu_userfunction_skepu_skel_0exposureFunc_SkePUExposureFuncCalc::CU(col[skepu_i * skepu_strides[0]], div);
		//skepu_output[skepu_i] = skepu_res;
		skepu_i += skepu_gridSize;
	}

	while (skepu_i < skepu_n)
	{
		
		auto skepu_tempMap = skepu_userfunction_skepu_skel_0exposureFunc_SkePUExposureFuncCalc::CU(col[skepu_i * skepu_strides[0]], div);
		skepu_result = skepu_userfunction_skepu_skel_0exposureFunc_add_float::CU(skepu_result, skepu_tempMap);
		skepu_i += skepu_gridSize;
	}

	sdata_skepu_skel_0[skepu_tid] = skepu_result;
	__syncthreads();

	if (skepu_blockSize >= 1024) { if (skepu_tid < 512 && skepu_tid + 512 < skepu_n) { sdata_skepu_skel_0[skepu_tid] = skepu_userfunction_skepu_skel_0exposureFunc_add_float::CU(sdata_skepu_skel_0[skepu_tid], sdata_skepu_skel_0[skepu_tid + 512]); } __syncthreads(); }
	if (skepu_blockSize >=  512) { if (skepu_tid < 256 && skepu_tid + 256 < skepu_n) { sdata_skepu_skel_0[skepu_tid] = skepu_userfunction_skepu_skel_0exposureFunc_add_float::CU(sdata_skepu_skel_0[skepu_tid], sdata_skepu_skel_0[skepu_tid + 256]); } __syncthreads(); }
	if (skepu_blockSize >=  256) { if (skepu_tid < 128 && skepu_tid + 128 < skepu_n) { sdata_skepu_skel_0[skepu_tid] = skepu_userfunction_skepu_skel_0exposureFunc_add_float::CU(sdata_skepu_skel_0[skepu_tid], sdata_skepu_skel_0[skepu_tid + 128]); } __syncthreads(); }
	if (skepu_blockSize >=  128) { if (skepu_tid <  64 && skepu_tid +  64 < skepu_n) { sdata_skepu_skel_0[skepu_tid] = skepu_userfunction_skepu_skel_0exposureFunc_add_float::CU(sdata_skepu_skel_0[skepu_tid], sdata_skepu_skel_0[skepu_tid +  64]); } __syncthreads(); }
	if (skepu_blockSize >=   64) { if (skepu_tid <  32 && skepu_tid +  32 < skepu_n) { sdata_skepu_skel_0[skepu_tid] = skepu_userfunction_skepu_skel_0exposureFunc_add_float::CU(sdata_skepu_skel_0[skepu_tid], sdata_skepu_skel_0[skepu_tid +  32]); } __syncthreads(); }
	if (skepu_blockSize >=   32) { if (skepu_tid <  16 && skepu_tid +  16 < skepu_n) { sdata_skepu_skel_0[skepu_tid] = skepu_userfunction_skepu_skel_0exposureFunc_add_float::CU(sdata_skepu_skel_0[skepu_tid], sdata_skepu_skel_0[skepu_tid +  16]); } __syncthreads(); }
	if (skepu_blockSize >=   16) { if (skepu_tid <   8 && skepu_tid +   8 < skepu_n) { sdata_skepu_skel_0[skepu_tid] = skepu_userfunction_skepu_skel_0exposureFunc_add_float::CU(sdata_skepu_skel_0[skepu_tid], sdata_skepu_skel_0[skepu_tid +   8]); } __syncthreads(); }
	if (skepu_blockSize >=    8) { if (skepu_tid <   4 && skepu_tid +   4 < skepu_n) { sdata_skepu_skel_0[skepu_tid] = skepu_userfunction_skepu_skel_0exposureFunc_add_float::CU(sdata_skepu_skel_0[skepu_tid], sdata_skepu_skel_0[skepu_tid +   4]); } __syncthreads(); }
	if (skepu_blockSize >=    4) { if (skepu_tid <   2 && skepu_tid +   2 < skepu_n) { sdata_skepu_skel_0[skepu_tid] = skepu_userfunction_skepu_skel_0exposureFunc_add_float::CU(sdata_skepu_skel_0[skepu_tid], sdata_skepu_skel_0[skepu_tid +   2]); } __syncthreads(); }
	if (skepu_blockSize >=    2) { if (skepu_tid <   1 && skepu_tid +   1 < skepu_n) { sdata_skepu_skel_0[skepu_tid] = skepu_userfunction_skepu_skel_0exposureFunc_add_float::CU(sdata_skepu_skel_0[skepu_tid], sdata_skepu_skel_0[skepu_tid +   1]); } __syncthreads(); }

	if (skepu_tid == 0)
		skepu_output[blockIdx.x] = sdata_skepu_skel_0[skepu_tid];
}

__global__ void skepu_skel_0_SkePURenderers_MapReduceKernel_SkePUExposureFuncCalc_add_float_ReduceOnly(float *skepu_input, float *skepu_output, size_t skepu_n, size_t skepu_blockSize, bool skepu_nIsPow2)
{
	extern __shared__ float sdata_skepu_skel_0[];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	size_t skepu_tid = threadIdx.x;
	size_t skepu_i = blockIdx.x * skepu_blockSize*2 + threadIdx.x;
	size_t skepu_gridSize = skepu_blockSize * 2 * gridDim.x;
	float skepu_result;

	if(skepu_i < skepu_n)
	{
		skepu_result = skepu_input[skepu_i];
		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		//This nIsPow2 opt is not valid when we use this kernel for sparse matrices as well where we
		// dont exactly now the elements when calculating thread- and block-size and nIsPow2 assum becomes invalid in some cases there which results in sever problems.
		// There we pass it always false
		if (skepu_nIsPow2 || skepu_i + skepu_blockSize < skepu_n)
			skepu_result = skepu_userfunction_skepu_skel_0exposureFunc_add_float::CU(skepu_result, skepu_input[skepu_i + skepu_blockSize]);
		skepu_i += skepu_gridSize;
	}

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a lParamer gridSize and therefore fewer elements per thread
	while(skepu_i < skepu_n)
	{
		skepu_result = skepu_userfunction_skepu_skel_0exposureFunc_add_float::CU(skepu_result, skepu_input[skepu_i]);
		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		if (skepu_nIsPow2 || skepu_i + skepu_blockSize < skepu_n)
			skepu_result = skepu_userfunction_skepu_skel_0exposureFunc_add_float::CU(skepu_result, skepu_input[skepu_i + skepu_blockSize]);
		skepu_i += skepu_gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata_skepu_skel_0[skepu_tid] = skepu_result;
	__syncthreads();

	// do reduction in shared mem
	if (skepu_blockSize >= 1024) { if (skepu_tid < 512) { sdata_skepu_skel_0[skepu_tid] = skepu_result = skepu_userfunction_skepu_skel_0exposureFunc_add_float::CU(skepu_result, sdata_skepu_skel_0[skepu_tid + 512]); } __syncthreads(); }
	if (skepu_blockSize >=  512) { if (skepu_tid < 256) { sdata_skepu_skel_0[skepu_tid] = skepu_result = skepu_userfunction_skepu_skel_0exposureFunc_add_float::CU(skepu_result, sdata_skepu_skel_0[skepu_tid + 256]); } __syncthreads(); }
	if (skepu_blockSize >=  256) { if (skepu_tid < 128) { sdata_skepu_skel_0[skepu_tid] = skepu_result = skepu_userfunction_skepu_skel_0exposureFunc_add_float::CU(skepu_result, sdata_skepu_skel_0[skepu_tid + 128]); } __syncthreads(); }
	if (skepu_blockSize >=  128) { if (skepu_tid <  64) { sdata_skepu_skel_0[skepu_tid] = skepu_result = skepu_userfunction_skepu_skel_0exposureFunc_add_float::CU(skepu_result, sdata_skepu_skel_0[skepu_tid +  64]); } __syncthreads(); }

	if (skepu_tid < 32)
	{
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		// UPDATE: volatile causes issues with custom struct data types; use __syncwarp() instead
		/*volatile*/ float* skepu_smem = sdata_skepu_skel_0;
		if (skepu_blockSize >=  64) { if (skepu_tid < 32) { skepu_smem[skepu_tid] = skepu_result = skepu_userfunction_skepu_skel_0exposureFunc_add_float::CU(skepu_result, skepu_smem[skepu_tid + 32]); } __syncwarp(); }
		if (skepu_blockSize >=  32) { if (skepu_tid < 16) { skepu_smem[skepu_tid] = skepu_result = skepu_userfunction_skepu_skel_0exposureFunc_add_float::CU(skepu_result, skepu_smem[skepu_tid + 16]); } __syncwarp(); }
		if (skepu_blockSize >=  16) { if (skepu_tid <  8) { skepu_smem[skepu_tid] = skepu_result = skepu_userfunction_skepu_skel_0exposureFunc_add_float::CU(skepu_result, skepu_smem[skepu_tid +  8]); } __syncwarp(); }
		if (skepu_blockSize >=   8) { if (skepu_tid <  4) { skepu_smem[skepu_tid] = skepu_result = skepu_userfunction_skepu_skel_0exposureFunc_add_float::CU(skepu_result, skepu_smem[skepu_tid +  4]); } __syncwarp(); }
		if (skepu_blockSize >=   4) { if (skepu_tid <  2) { skepu_smem[skepu_tid] = skepu_result = skepu_userfunction_skepu_skel_0exposureFunc_add_float::CU(skepu_result, skepu_smem[skepu_tid +  2]); } __syncwarp(); }
		if (skepu_blockSize >=   2) { if (skepu_tid <  1) { skepu_smem[skepu_tid] = skepu_result = skepu_userfunction_skepu_skel_0exposureFunc_add_float::CU(skepu_result, skepu_smem[skepu_tid +  1]); } __syncwarp(); }
	}

	// write result for this block to global mem
	if (skepu_tid == 0)
		skepu_output[blockIdx.x] = sdata_skepu_skel_0[0];
}
