
__global__ void skepu_skel_0_SkePUDenoiserNN_MapKernel_SkePUBPFunc(SkePUBPOut* skepu_output, skepu::PRNG::Placeholder,SkePUBPIn *f, int samples, float learningRate,  size_t skepu_w2, size_t skepu_w3, size_t skepu_w4, size_t skepu_n, size_t skepu_base, skepu::StrideList<2> skepu_strides)
{
	size_t skepu_i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t skepu_global_prng_id = skepu_i;
	size_t skepu_gridSize = blockDim.x * gridDim.x;
	
	if (skepu_strides[0] < 0) { skepu_output += (-skepu_n + 1) * skepu_strides[0]; }
if (skepu_strides[1] < 0) { f += (-skepu_n + 1) * skepu_strides[1]; }


	while (skepu_i < skepu_n)
	{
		
		
		auto skepu_res = skepu_userfunction_skepu_skel_0map_SkePUBPFunc::CU(f[skepu_i * skepu_strides[1]], samples, learningRate);
		skepu_output[skepu_i * skepu_strides[0]] = skepu_res;
		skepu_i += skepu_gridSize;
	}
}
