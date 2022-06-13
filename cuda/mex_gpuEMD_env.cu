// Modification: 2022/Feb/25
// Silent the code
// Modify the kernal for finding extrema of the resultant IMFs (kernel name: extreme_amp)
// So that the output upp is the same as using instananeous amplitude in holo
// Apply extreme_amp to every IMF, and remove lower envelope from output.
// ======================================================
        
// Do not change the number of threads, blocks!!!!!
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdlib>
#include <builtin_types.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
// #include <unistd.h>
#include <stddef.h>
// #include <dirent.h>
#include <time.h>
//#include "common.h"
//#include "file.h"
//#include "golden.cpp"
//#include "wrapper.cu"
        
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "matrix.h"
#include "emd_kernels_extreme_amp.cu"

#define D1_DEBUG 0
//#define MAX_ITERATION 10 // number of sifting
#define SEGMENT_BYTES 128
#define WRAPPER_BACK 0
//#define MEMORY_MAX_SIZE 1024*100 //byts, should be an device-dependent

        
void pure_emd(
    double *h_imf,      //final imf, should be 1 dimensional, (x_len * y_len * tnm)
    double *h_value,    //critical points (value) of each imf
    double *h_index,    //critical points (index) of each imf
    int *h_len,         //length of critical points of each imf
    double *h_upper,    //upper envelop of each imf
    //double *h_lower,    //lower envelop of each imf
	const double *s_data,   //input signal, should be 2 dimensional, (x_len, y_len)
	const int x_len,
	const int y_len,
    const int tnm,
    const int MAX_ITERATION
)
{
	printf("Running parallele EMD on GPU .................................................\n");
	//double *d_final;    //final imf, should be 1 dimensional, (x_len * y_len * tnm)
    double *d_data;     //input signal, should be 2 dimensional, (x_len, y_batch_len)
    //double *d_crit_pt_allimf;
    //double *d_amp_allimf;
    ///partial buffer (on device)
    double *d_current;  //proto-imf during sifting, should be 2 dimensional, (x_len, y_batch_length)
	double *d_max_value;
	double *d_min_value;
	int *d_max_index;
	int *d_min_index;
	int *d_max_len;
	int *d_min_len;
	double *d_max_y2;
	double *d_min_y2;
	double *d_max_u;
    double *d_min_u;

    double *d_value;
    double *d_index;
    int *d_len;
	double *d_upper;
    double *d_lower;
    
    
	//Get device property
    int deviceCount;
    cudaDeviceProp deviceProp;
    size_t free_t,total_t;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaGetDeviceCount(&deviceCount);
    cudaMemGetInfo(&free_t,&total_t);
       
    //printf("device Count:%d\n",deviceCount+1);
    //printf("GPU device: %s\n",deviceProp.name);
    //printf("Max threads per block: %d\n",deviceProp.maxThreadsPerBlock);
    //printf("Available memory : %zu bits\n",free_t);
    int MEMORY_MAX_SIZE = free_t/1024;
    

	//calculate padding
    //int align_len=SEGMENT_BYTES/sizeof(double);
    

	int align_len=16; // should be consistent with thread_back
    int x_padding=((x_len+align_len-1)/align_len)*align_len;
    //printf("> x_len is %d, y_len is %d\n", x_len, y_len);
	//printf("> x_padding is %d\n",x_padding);
    
    //  ==  Calculate maximum allowable size for batch ==
    // If the data is too large, we cut the y_len into several batches,
    // y_batch_len is the maximum y length in each batch 
    // Each batch needs the following memory for all steps during sifting:
    // variable                |  size
    // ------------------------+----------------------------------
    // d_data, d_current       | x_padding * double * y_batch_len (8 bytes * 1)
    // d_max_val, d_min_val    | x_padding * double * y_batch_len (8 bytes * 2)
    // d_max_index,d_min_index | x_padding * int    * y_batch_len (4 bytes * 2)
    // d_max_len, d_min_len    |     1     * int    * y_batch_len
    // d_max_y2, d_min_y2      | x_padding * int    * y_batch_len (8 bytes * 2)
    // d_max_u,  d_min_u       | x_padding * int    * y_batch_len (8 bytes * 2)
    // ------------------------+------------------------------
    // Total = (x_padding + 2)* 72 bytes * y_batch_len
    // So, the y_batch_len is MEMORY_MAX_SIZE/72/ (x_padding +2)
    // For each batch

    //int y_align_len = THREAD_MAIN;  // Defined in emd_kernels; = 64;
    int y_align_len = 256;  // Defined in emd_kernels; = 64;
    int y_batch_len = ((y_len + y_align_len -1)/y_align_len)*y_align_len;
    //int y_batch_len_max=(MEMORY_MAX_SIZE*1024/72/(x_padding+2)/y_align_len)*y_align_len;
    int y_batch_len_max=(free_t/72/(x_padding+2)/y_align_len)*y_align_len;
    y_batch_len=y_batch_len<y_batch_len_max?y_batch_len:y_batch_len_max;
	int y_batch_times=(y_len+y_batch_len-1)/y_batch_len;

    //printf("> THREAD_MAIN is %d\n", THREAD_MAIN);
    //printf("> MEMORY_MAX_SIZE is %d bytes\n", MEMORY_MAX_SIZE);
    //printf("> Maximum allowable y length is %d\n", y_batch_len_max);
    //printf("> y_batch_len is %d\n",y_batch_len);
    //printf("> y_batch_times is %d\n",y_batch_times);

	
	
	double total_size=0.0;
	int size;
	
	//b_malloc=get_mytime ();
	//data malloc and memset
	{
        // d_data (input)
        size=x_padding*y_batch_len*sizeof(double);
        cudaMalloc((void **)&d_data, size); 
        cudaMemset(d_data,0, x_padding*y_batch_len*sizeof(double)); 
        total_size+=size;
        // d_current (temperary imf)
        cudaMalloc((void **)&d_current, size); 
        cudaMemset(d_current,0, x_padding*y_batch_len*sizeof(double));
        total_size+=size;
        // d_final (final imf)
        //size=tnm*x_padding*y_batch_len*sizeof(double);
        //cudaMalloc((void **)&d_final, size); 
        //cudaMemset(d_imf,0, x_padding*y_batch_len*sizeof(double));
        //total_size+=size;

        // partial memloc
        // d_max_value & d_min_value
        size=x_padding*y_batch_len*sizeof(double);
        cudaMalloc((void **)&d_max_value, size); 
        cudaMemset(d_max_value,0, size);
        cudaMalloc((void **)&d_min_value, size);
        cudaMemset(d_min_value,0, size);
        total_size+=size;        total_size+=size;
        // d_max_index & d_min_index
        size=x_padding*y_batch_len*sizeof(int);
        cudaMalloc((void **)&d_max_index, size); 
        cudaMemset(d_max_index,0, size);
        cudaMalloc((void **)&d_min_index, size);
        cudaMemset(d_min_index,0, size);
        total_size+=size;        total_size+=size;

        // d_max_len & d_min_len
        size=y_batch_len*sizeof(int);
        cudaMalloc((void **)&d_max_len, size); 
        cudaMemset(d_max_len,0, size);
        cudaMalloc((void **)&d_min_len, size);
        cudaMemset(d_min_len,0, size);
        total_size+=size;        total_size+=size;

        // d_max_y2, d_min_y2, d_max_u, d_min_u
        size=x_padding*y_batch_len* sizeof(double);
        cudaMalloc((void **)&d_max_y2, size); 		
        cudaMalloc((void **)&d_min_y2, size);
        cudaMalloc((void **)&d_max_u, size); 		
        cudaMalloc((void **)&d_min_u, size);
        cudaMemset(d_max_y2,0, size); 
        cudaMemset(d_min_y2,0, size); 
        cudaMemset(d_max_u,0, size); 
        cudaMemset(d_min_u,0, size); 
        total_size+=size;        total_size+=size;
        total_size+=size;        total_size+=size;

        // d_value
        size=2*x_padding*y_batch_len*sizeof(double);
        cudaMalloc((void **)&d_value, size); 
        cudaMemset(d_value,0, size);
        total_size+=size;
        // d_len
        size=y_batch_len*sizeof(int);
        cudaMalloc((void **)&d_len, size); 
        cudaMemset(d_len,0, size);
        total_size+=size;
        // d_index
        size=2*x_padding*y_batch_len*sizeof(double);
        cudaMalloc((void **)&d_index, size); 
        cudaMemset(d_index,0, size);
        total_size+=size;
        // d_upper
        size=x_padding*y_batch_len*sizeof(double);
        cudaMalloc((void **)&d_upper, size); 
        cudaMemset(d_upper,0, x_padding*y_batch_len*sizeof(double));
        total_size+=size;
        // d_lower
        size=x_padding*y_batch_len*sizeof(double);
        cudaMalloc((void **)&d_lower, size); 
        cudaMemset(d_lower,0, x_padding*y_batch_len*sizeof(double)); 
        total_size+=size;
	}
	
	
	//printf("> Used memory size: %f MB \n",  total_size/1024.0f/1024.0f);

	//printf("> cudaMemset done\n");

	
	cudaFuncSetCacheConfig(extreme,cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(forward,cudaFuncCachePreferShared);
	cudaFuncSetCacheConfig(backward,cudaFuncCachePreferL1);
	cudaFuncSetCacheConfig(spline_interp_0,cudaFuncCachePreferL1);
 	cudaFuncSetCacheConfig(spline_interp,cudaFuncCachePreferL1);
 	cudaFuncSetCacheConfig(update_x0,cudaFuncCachePreferShared);
    //printf("> cudaFuncSetCasheConfig done, line 176 \n");
	
    int jump = 1;
    int bias;
	int high_stride = x_padding;
    double *h_imf_ptr;
    int y_len_active;
    //b_gpu=get_mytime();
    


    // for each batch (y dimemsion)
	for(int b=0;b<y_batch_times;b++)
	{
        bias=b*y_batch_len; //number of rows jumped by this block
		int last_len=y_len-bias;
		y_len_active=last_len>y_batch_len?y_batch_len:last_len;

		///dim3 grid_ens ((x_len+EMD_EXPAND_SHARED_SIZE-1)/EMD_EXPAND_SHARED_SIZE , y_len_active );
		///ensemble_expansion_lowbit<<<grid_ens , ne >>>(d_ensemble,data_ptr,d_noise, x_padding ,x_len);
        
        //b_memcpy=get_mytime ();
        //memcpy		
	    {
		    for(int i=0;i<y_len_active;i++)
		    {
			    cudaMemcpy(d_data+i*x_padding, s_data+(bias+i)*x_len, x_len*sizeof(double), cudaMemcpyHostToDevice);
		    }
        }
        
	    //e_memcpy=get_mytime();
	    //s_memcpy+=e_memcpy-b_memcpy;
        //printf("> cudaMemcpy to host done, line 201 \n");
        
        // pure EMD here
        int thread_low_forward=16;
        dim3 grid_low_forward((y_batch_len+thread_low_forward-1)/thread_low_forward);
        dim3 thread_back(16,16);
        dim3 grid_back(y_batch_len/EMD_BACKWARD_SHARED_SIZE);
        dim3 thread_update = 16;
        dim3 grid_update((y_batch_len+thread_update.x-1)/thread_update.x);
			
        int k=0;
		for(k=0;k<tnm-1;k++)
		{	
            
            //cudaThreadSynchronize();
            //printf("> pureEMD running on batch = %d, imf = %d\n", b, k);
            // define thread, grid dimensions
			
// 			dim3 grid_low_forward((y_len_active+thread_low_forward-1)/thread_low_forward);

//             dim3 grid_back(y_len_active/EMD_BACKWARD_SHARED_SIZE);
            // [First sifting]
            //    extrema: work on d_data, 
            //    after sifting: d_current = d_data - spline_trend
			{
            extreme<<< grid_low_forward, thread_low_forward>>>(d_max_value, d_min_value,d_max_index, d_min_index,d_max_len,d_min_len,d_data, x_len, high_stride, jump);
            forward<<< grid_low_forward, thread_low_forward>>>(d_max_y2, d_min_y2,d_max_u, d_min_u, d_max_value, d_min_value,d_max_index, d_min_index, d_max_len, d_min_len, high_stride,jump);
            backward<<<  grid_back, thread_back>>>(d_max_y2, d_min_y2,d_max_u, d_min_u,d_max_len,d_min_len, high_stride,jump);
            //int thread_spline=THREAD_MAIN;
            //dim3 grid_spline(2*ne/thread_spline,y_len_active);
            spline_interp_0<<< grid_low_forward, thread_low_forward>>>(d_current, d_data, d_max_y2, d_min_y2,d_max_value, d_min_value, d_max_index,d_min_index, x_len, high_stride,jump);
            }

            // [Second and later siftings]
            //    extrema: work on d_current, 
            //    after sifting: d_current = d_current - spline_trend
			for(int i=1;i<MAX_ITERATION;i++)
			{
                extreme<<< grid_low_forward, thread_low_forward>>>(d_max_value, d_min_value,d_max_index, d_min_index,d_max_len,d_min_len,d_current, x_len, high_stride,jump);
                forward<<< grid_low_forward, thread_low_forward>>>(d_max_y2, d_min_y2,d_max_u, d_min_u, d_max_value, d_min_value,d_max_index, d_min_index,d_max_len,d_min_len,high_stride,jump);
                //	extreme_forward<<< grid_low_forward, thread_low_forward>>>(max_y2, min_y2,max_u, min_u,max_value, min_value,max_index, min_index,max_len,min_len,d_current, x_len, 2*ne*x_len,jump);
                ///dim3 thread_back(16,16);
                ///dim3 grid_back(y_batch_len/EMD_BACKWARD_SHARED_SIZE);
                backward<<<  grid_back, thread_back>>>(d_max_y2,d_min_y2,d_max_u,d_min_u,d_max_len,d_min_len, high_stride,jump);
                spline_interp<<< grid_low_forward, thread_low_forward>>>(d_current,d_max_y2, d_min_y2,d_max_value, d_min_value, d_max_index,d_min_index, x_len, high_stride,jump);
			}
            // now, after MAX_ITERATION, d_current is the kth IMF
            

            // ====================================================
            // Next, save the IMF and update d_data
            // (1) copy the IMF (d_current) from device to host (h_imf)
            //h_imf_ptr = h_imf + k*x_len*y_len + bias*x_len;
            for(int i=0;i<y_len_active;i++)
            {
                cudaMemcpy(h_imf + k*x_len*y_len + bias*x_len + i*x_len, d_current+i*x_padding, x_len*sizeof(double), cudaMemcpyDeviceToHost);
            }


            
            // (2) update d_data as the next x_0 (d_data = d_data - d_current)
			//if(k<tnm-2){
            update_x0<<<grid_update , thread_update>>>(d_data, d_current, high_stride ,x_len, jump);
            //}
            
            // (3) calculate critical points and amplitude for this imf (d_current)
            // (3.1) critical_pts for the imf (d_current)
            critical<<<grid_low_forward, thread_low_forward>>>(d_value, d_index, d_len, d_current, x_len, high_stride, jump);
            // (3.2) amplitude (abs) for the imf
            extreme_amp<<< grid_low_forward, thread_low_forward>>>(d_max_value, d_min_value,d_max_index, d_min_index,d_max_len,d_min_len, d_current, x_len, high_stride, jump);
            forward<<< grid_low_forward, thread_low_forward>>>(d_max_y2, d_min_y2,d_max_u, d_min_u, d_max_value, d_min_value,d_max_index, d_min_index, d_max_len, d_min_len, high_stride,jump);
            backward<<<  grid_back, thread_back>>>(d_max_y2, d_min_y2,d_max_u, d_min_u,d_max_len,d_min_len, high_stride,jump);
            spline_interp_uplow<<< grid_low_forward, thread_low_forward>>>(d_upper, d_lower, d_current, d_max_y2, d_min_y2,d_max_value, d_min_value, d_max_index,d_min_index, x_len, high_stride,jump);
            
            //bias=b*y_batch_len; 
            //cudaMemcpy(h_imf + k*x_len*y_len + bias*x_len + i*x_len, d_current+i*x_padding, x_len*sizeof(double), cudaMemcpyDeviceToHost);
            for(int i=0;i<y_len_active;i++)
            {
            // (3.3) cpy critical_pts back to host
                cudaMemcpy(h_value + k*2*x_len*y_len + bias*2*x_len + i*2*x_len, d_value+i*2*x_padding, 2*x_len*sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_index + k*2*x_len*y_len + bias*2*x_len + i*2*x_len, d_index+i*2*x_padding, 2*x_len*sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_len + k*y_len + bias +i, d_len+i, sizeof(int), cudaMemcpyDeviceToHost);
            // (3.4) cpy amplitude back to host
                cudaMemcpy(h_upper + k*x_len*y_len + bias*x_len + i*x_len, d_upper+i*x_padding, x_len*sizeof(double), cudaMemcpyDeviceToHost);
                //cudaMemcpy(h_lower + k*x_len*y_len + bias*x_len + i*x_len, d_lower+i*x_padding, x_len*sizeof(double), cudaMemcpyDeviceToHost);
            }
            
        } //End of for(k<tnm)
        //printf("> end of for (k<tnm), k is now %d\n", k);
        
        // copy the last mode (d_data) to h_imf
        h_imf_ptr = h_imf + (k)*x_len*y_len + bias*x_len;
        for(int i=0;i<y_len_active;i++)
        {
            cudaMemcpy(h_imf_ptr + i*x_len, d_data+i*x_padding, x_len*sizeof(double), cudaMemcpyDeviceToHost);
//            cudaMemcpy(h_imf_ptr + i*x_len, d_current+i*x_padding, x_len*sizeof(double), cudaMemcpyDeviceToHost);
        }
        
        // (3) calculate critical points and amplitude for the last imf (d_data)
                // (3.1) critical_pts for the imf (d_current)
                critical<<<grid_low_forward, thread_low_forward>>>(d_value, d_index, d_len, d_data, x_len, high_stride, jump);

                // (3.2) amplitude for the imf
                extreme_amp<<< grid_low_forward, thread_low_forward>>>(d_max_value, d_min_value,d_max_index, d_min_index,d_max_len,d_min_len,d_data, x_len, high_stride, jump);
                forward<<< grid_low_forward, thread_low_forward>>>(d_max_y2, d_min_y2,d_max_u, d_min_u, d_max_value, d_min_value,d_max_index, d_min_index, d_max_len, d_min_len, high_stride,jump);
                backward<<<  grid_back, thread_back>>>(d_max_y2, d_min_y2,d_max_u, d_min_u,d_max_len,d_min_len, high_stride,jump);
                spline_interp_uplow<<< grid_low_forward, thread_low_forward>>>(d_upper, d_lower, d_data, d_max_y2, d_min_y2,d_max_value, d_min_value, d_max_index,d_min_index, x_len, high_stride,jump);
                
                
                //bias=b*y_batch_len; 
                //cudaMemcpy(h_imf + k*x_len*y_len + bias*x_len + i*x_len, d_current+i*x_padding, x_len*sizeof(double), cudaMemcpyDeviceToHost);
                for(int i=0;i<y_len_active;i++)
                {
                // (3.3) cpy critical_pts back to host
                    cudaMemcpy(h_value + k*2*x_len*y_len + bias*2*x_len + i*2*x_len, d_value+i*2*x_padding, 2*x_len*sizeof(double), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_index + k*2*x_len*y_len + bias*2*x_len + i*2*x_len, d_index+i*2*x_padding, 2*x_len*sizeof(double), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_len + k*y_len + bias +i, d_len+i, sizeof(int), cudaMemcpyDeviceToHost);
                // (3.4) cpy amplitude back to host

                    cudaMemcpy(h_upper + k*x_len*y_len + bias*x_len + i*x_len, d_upper+i*x_padding, x_len*sizeof(double), cudaMemcpyDeviceToHost);
                    //cudaMemcpy(h_lower + k*x_len*y_len + bias*x_len + i*x_len, d_lower+i*x_padding, x_len*sizeof(double), cudaMemcpyDeviceToHost);
                }
        
	} //End of b<y_batch_times

	printf("> GPU EMD completed\n");
    
	
	//cudaThreadSynchronize();
	//e_gpu=get_mytime();
	//s_gpu+=e_gpu-b_gpu;
    

	//printf("> GPU malloc time is %lf \n",s_malloc);
	//printf("> GPU random time is %lf \n",s_random);
	//printf("> GPU memcpy time is %lf \n",s_memcpy);
	//printf("> GPU execution time is %lf \n",s_gpu);
	
	
	{
	//cudaFree(d_final);
	cudaFree(d_data);
	cudaFree(d_max_value);
	cudaFree(d_min_value);
	cudaFree(d_max_index);
	cudaFree(d_min_index);
	cudaFree(d_max_len);
	cudaFree(d_min_len);
	cudaFree(d_max_y2);
	cudaFree(d_min_y2);
	cudaFree(d_max_u);
	cudaFree(d_min_u);
	//cudaFree(d_ensemble);
    cudaFree(d_current);
    
    cudaFree(d_value);
	cudaFree(d_index);
    cudaFree(d_len);
	cudaFree(d_upper);
	cudaFree(d_lower);
    }


}

// Entry point for Matlab
// The matlab function:
//   all_imfs = mex_pure_cuEMD(input, nm)
// The main function above: 
// void pure_emd(
//	  double *h_imf,          //final imf, should be 1 dimensional, (x_len * y_len * tnm)
//	  const double *s_data,   //input signal, should be 2 dimensional, (x_len, y_len)
//	  const int x_len,
//	  const int y_len,
//    const int tnm)

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* input matrix */
    double *inMatrix = mxGetPr(prhs[0]);
    int nm = (int) *mxGetPr(prhs[1]);  // Better: *mxGetData(prhs[3])
    int nsift = (int) *mxGetPr(prhs[2]);  // Better: *mxGetData(prhs[3])
    mexPrintf("Number of IMF = %i\n", nm);
    /* output matrix */
    double *outMatrix;
    double *upMatrix;
    //double *lowMatrix;
    double *h_val;
    double *h_idx;         
    //double *h_max_val, *h_min_val, *h_zero_val;
    //double *h_zero_idx;
    //int *h_max_idx, *h_min_idx;
    //int *h_max_len, *h_min_len;

    /* get dimensions of the input matrix */
    int x_len, y_len;
    x_len = mxGetM(prhs[0]);
    y_len = mxGetN(prhs[0]);

    //int x_len = (int) *mxGetPr(prhs[2]);
    //int y_len = (int) *mxGetPr(prhs[3]);
    mexPrintf("x_len value is %d, y_len is %d\n", x_len, y_len);


    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix(x_len*y_len*nm,1,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(2*x_len*y_len*nm,1, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(2*x_len*y_len*nm,1, mxREAL);
    plhs[3] = mxCreateNumericMatrix(y_len, nm, mxINT32_CLASS, mxREAL); //h_len
    plhs[4] = mxCreateDoubleMatrix(x_len*y_len*nm,1,mxREAL); //upper
    //plhs[5] = mxCreateDoubleMatrix(x_len*y_len*nm,1,mxREAL); //lower
    /*

    plhs[4] = mxCreateNumericMatrix(x_len,y_len, mxINT32_CLASS, mxREAL);
    plhs[5] = mxCreateNumericMatrix(x_len,y_len, mxINT32_CLASS, mxREAL);
    plhs[6] = mxCreateNumericMatrix(x_len,y_len, mxDOUBLE_CLASS, mxREAL);
    plhs[7] = mxCreateNumericMatrix(1,y_len, mxINT32_CLASS, mxREAL);
    */
    /* get a pointer to the real data in the output matrix */
    outMatrix = mxGetPr(plhs[0]);
    h_val = mxGetPr(plhs[1]);
    h_idx = mxGetPr(plhs[2]);
    int *h_len = (int *) mxGetData(plhs[3]);
    upMatrix = mxGetPr(plhs[4]);
    //lowMatrix = mxGetPr(plhs[5]);
    /*
    h_max_val = mxGetPr(plhs[1]);
    h_min_val = mxGetPr(plhs[2]);
    h_zero_val =  mxGetPr(plhs[3]);
    int *h_max_idx = (int *) mxGetData(plhs[4]);
    int *h_min_idx = (int *) mxGetData(plhs[5]);
    h_zero_idx = mxGetPr(plhs[6]);
    int *h_max_len = (int *) mxGetData(plhs[7]);
    int *h_min_len = (int *) mxGetData(plhs[8]);
    int *h_zero_len = (int *) mxGetData(plhs[9]);
    */

    /* call the computational routine */
    //mainfun(inMatrix,x_len,y_len,nm,outMatrix,h_max_val,h_min_val,h_zero_val,h_max_idx, h_min_idx,h_zero_idx, h_max_len, h_min_len,h_zero_len);
    pure_emd(outMatrix,h_val,h_idx,h_len,upMatrix,inMatrix, x_len, y_len, nm, nsift);
}