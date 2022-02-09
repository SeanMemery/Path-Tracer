
__global__ void SkePUDenoiserNN_Overlap2DKernel_SkePUFPFunc_conv_cuda_2D_kernel(ForPropOut* skepu_output, skepu::PRNG::Placeholder,struct ForPropIn *skepu_input, SkePUFPConstants sConstants, 
	const size_t skepu_in_rows, const size_t skepu_in_cols,
	const size_t skepu_out_rows, const size_t skepu_out_cols,
	size_t skepu_overlap_y, size_t skepu_overlap_x,
	size_t skepu_in_pitch, size_t skepu_out_pitch,
	const size_t skepu_sharedRows, const size_t skepu_sharedCols,
	skepu::Edge skepu_edge, struct ForPropIn skepu_pad
)
{
  extern __shared__ struct ForPropIn sdata_skepu_skel_2[];
	size_t skepu_xx = blockIdx.x * blockDim.x;
	size_t skepu_yy = blockIdx.y * blockDim.y;

	size_t skepu_x = skepu_xx + threadIdx.x;
	size_t skepu_y = skepu_yy + threadIdx.y;
	
	
	if (skepu_x < skepu_out_cols + skepu_overlap_x * 2 && skepu_y < skepu_out_rows + skepu_overlap_y * 2)
	{
		size_t skepu_shared_x = threadIdx.x;
		size_t skepu_shared_y = threadIdx.y;
		while (skepu_shared_y < skepu_sharedRows)
		{
			while (skepu_shared_x < skepu_sharedCols)
			{
				size_t skepu_sharedIdx = skepu_shared_y * skepu_sharedCols + skepu_shared_x;
				int skepu_global_x = (skepu_xx + skepu_shared_x - skepu_overlap_x);
				int skepu_global_y = (skepu_yy + skepu_shared_y - skepu_overlap_y);
				
				if ((skepu_global_y >= 0 && skepu_global_y < skepu_in_rows) && (skepu_global_x >= 0 && skepu_global_x < skepu_in_cols))
					sdata_skepu_skel_2[skepu_sharedIdx] = skepu_input[skepu_global_y * skepu_in_cols + skepu_global_x];
				else
				{
					if (skepu_edge == skepu::Edge::Pad)
						sdata_skepu_skel_2[skepu_sharedIdx] = skepu_pad;
					else if (skepu_edge == skepu::Edge::Duplicate)
					{
						sdata_skepu_skel_2[skepu_sharedIdx] = skepu_input[
							skepu::cuda::clamp(skepu_global_y, 0, (int)skepu_in_rows - 1) * skepu_in_cols +
							skepu::cuda::clamp(skepu_global_x, 0, (int)skepu_in_cols - 1)];
					}
					else if (skepu_edge == skepu::Edge::Cyclic)
					{
						sdata_skepu_skel_2[skepu_sharedIdx] = skepu_input[
							((skepu_global_y + skepu_in_rows) % skepu_in_rows) * skepu_in_cols +
							((skepu_global_x + skepu_in_cols) % skepu_in_cols)];
					}
				}
				
				skepu_shared_x += blockDim.x;
			}
			skepu_shared_x  = threadIdx.x;
			skepu_shared_y += blockDim.y;
		}
	}

	__syncthreads();
	
	

	if (skepu_x < skepu_out_cols && skepu_y < skepu_out_rows)
	{
		size_t skepu_w2 = skepu_out_cols;
		size_t skepu_i = skepu_y * skepu_out_cols + skepu_x;
		size_t skepu_global_prng_id = skepu_i;
		size_t skepu_base = 0;
		
		
		auto skepu_res = skepu_userfunction_skepu_skel_2convol_SkePUFPFunc::CU({(int)skepu_overlap_y, (int)skepu_overlap_x, skepu_sharedCols, &sdata_skepu_skel_2[(threadIdx.y + skepu_overlap_y) * skepu_sharedCols + (threadIdx.x + skepu_overlap_x)]}, sConstants);
		skepu_output[skepu_i] = skepu_res;
	}
}
