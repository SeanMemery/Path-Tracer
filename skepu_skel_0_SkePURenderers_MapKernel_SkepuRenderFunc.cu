
__global__ void skepu_skel_0_SkePURenderers_MapKernel_SkepuRenderFunc(ReturnStruct* skepu_output, skepu::PRNG::Placeholder,RandomSeeds *seeds, Constants constants,  size_t skepu_w2, size_t skepu_w3, size_t skepu_w4, size_t skepu_n, size_t skepu_base, skepu::StrideList<2> skepu_strides)
{
	size_t skepu_i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t skepu_global_prng_id = skepu_i;
	size_t skepu_gridSize = blockDim.x * gridDim.x;
	
	if (skepu_strides[0] < 0) { skepu_output += (-skepu_n + 1) * skepu_strides[0]; }
if (skepu_strides[1] < 0) { seeds += (-skepu_n + 1) * skepu_strides[1]; }


	while (skepu_i < skepu_n)
	{
		skepu::Index2D skepu_index;
skepu_index.row = (skepu_base + skepu_i) / skepu_w2;
skepu_index.col = (skepu_base + skepu_i) % skepu_w2;
		
		auto skepu_res = skepu_userfunction_skepu_skel_0renderFunc_SkepuRenderFunc::CU(skepu_index, seeds[skepu_i * skepu_strides[1]], constants);
		skepu_output[skepu_i * skepu_strides[0]] = skepu_res;
		skepu_i += skepu_gridSize;
	}
}
