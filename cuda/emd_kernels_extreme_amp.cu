// A 2-D image, say X[i,j] 1<=i<=H, 1<=j<=L
// H = y_len, L =x_len
// after zero-padding, x_dim_len becomes L_padding
// y_i represents each row of X, That is, X[i,]
// x_i represents each column of X, That is, X[,j]

#define STD_L_SHARED_SIZE 256
#define STD_H_SHARED_SIZE 256
#define EMD_BACKWARD_SHARED_SIZE 256
#define EMD_EXPAND_SHARED_SIZE 32
#define UPDATE_IMF_SHARED_SIZE 256
#define EPS 1e-6
#define THREAD_MAIN 64
__constant__ double CONST_STD[1024];

//////////////////////
// Calculate the std of each y_i; i.e. X[i,]
///  H x L_padding
///  threadIdx.x moves along L_padding (paralize computing within a signal)
///  blockIdx.x  moves along H
//////////////////////
__global__ void std_lowbit(double *std ,const double * signal,const int len,const int padding_len)
{
	const int tx=threadIdx.x;
	const int bx=blockIdx.x*padding_len;
	__shared__ double sh_x[STD_L_SHARED_SIZE];
	__shared__ double sh_x2[STD_L_SHARED_SIZE];
	
	
	//initialization
	sh_x[tx]=0.0;
	sh_x2[tx]=0.0;
	__syncthreads();
	
	//caculate address
	signal+=bx;
	
	for(int i=tx;i<len;i+=blockDim.x)
	{
		double x=signal[i];
		sh_x[tx]+=x;
		sh_x2[tx]+=x*x;
	}
	__syncthreads();
	
	//reduction
#if STD_L_SHARED_SIZE>=512
	if(tx<256) { sh_x[tx]+=sh_x[tx+256];
				 sh_x2[tx]+=sh_x2[tx+256];}  
	__syncthreads();
#endif	
	
#if STD_L_SHARED_SIZE>=256
	if(tx<128) { sh_x[tx]+=sh_x[tx+128];
				 sh_x2[tx]+=sh_x2[tx+128];}  
	__syncthreads();
#endif
	
#if STD_L_SHARED_SIZE>=128
	if(tx<64) { sh_x[tx]+=sh_x[tx+64];
				sh_x2[tx]+=sh_x2[tx+64];}	
	__syncthreads();
#endif
	

	if(tx<32) { sh_x[tx]+=sh_x[tx+32];
				sh_x2[tx]+=sh_x2[tx+32];}	
	//__syncthreads();
	if(tx<16) { sh_x[tx]+=sh_x[tx+16];
				sh_x2[tx]+=sh_x2[tx+16];}	
	//__syncthreads();
	if(tx<8) { sh_x[tx]+=sh_x[tx+8];
			   sh_x2[tx]+=sh_x2[tx+8];}	
	//__syncthreads();
	if(tx<4) { sh_x[tx]+=sh_x[tx+4];
			   sh_x2[tx]+=sh_x2[tx+4];}	
	//__syncthreads();
	if(tx<2) { sh_x[tx]+=sh_x[tx+2];
			   sh_x2[tx]+=sh_x2[tx+2];}	
	//__syncthreads();
	
	if(tx<1) { 
		double one_over_len=1.0/(double)len;
		sh_x[0]+=sh_x[tx+1];
		sh_x2[0]+=sh_x2[tx+1];
		sh_x[0]*=one_over_len;
		sh_x2[0]*=one_over_len;	
		std[blockIdx.x]=sqrt(sh_x2[0]-sh_x[0]*sh_x[0]);
	}	
	
}
//////////////////
///  H x L_padding
///  threadIdx.x is L_padding
///  threadIdx.y is H
///  blockIdx.x is L_padding
///  blockDim.y is H
//////////////////////
__global__ void std_highbit(
					double *std ,
					const double * signal,
					const int len,			// current dim len
					const int stride,		// stride between to element
					const int x_len // 
					)
{
	const int tx=threadIdx.x;
	const int ty=threadIdx.y;
	//const int bx=blockIdx.x*padding_len;
	
	__shared__ double sh_x[STD_H_SHARED_SIZE];  
	__shared__ double sh_x2[STD_H_SHARED_SIZE]; 
	const int id = ty*blockDim.x+tx;   // for shared memory
	sh_x[id]=0.0;
	sh_x2[id]=0.0;
	
	
	
	const int bx = blockIdx.x * blockDim.x;

	const int jump=stride*blockDim.y;
	std += bx+tx;
	__syncthreads();
	if( bx+tx < x_len)
	{
		const double * signal_end = signal+len*stride;
		signal += bx+tx + ty*stride;
		do
		{
			double x=signal[0]; signal+=jump;
			sh_x[id]+=x;
			sh_x2[id]+=x*x;
		}while(signal<signal_end);

	}
	__syncthreads();

	
	//reduction
#if STD_L_SHARED_SIZE>256
	if(id<256) { sh_x[id]+=sh_x[id+256];
				 sh_x2[id]+=sh_x2[id+256];}  
	__syncthreads();
#endif	
	
#if STD_L_SHARED_SIZE>128
	if(id<128) { sh_x[id]+=sh_x[id+128];
				 sh_x2[id]+=sh_x2[id+128];}  
	__syncthreads();
#endif
	
#if STD_L_SHARED_SIZE>64
	if(id<64) { sh_x[id]+=sh_x[id+64];
				sh_x2[id]+=sh_x2[id+64];}	
	__syncthreads();
#endif

#if STD_L_SHARED_SIZE>32
	if(id<32) { sh_x[id]+=sh_x[id+32];
				sh_x2[id]+=sh_x2[id+32];}	
#endif
	if(id<16) { 
		double one_over_len=1.0/(double)len;
		sh_x[id]+=sh_x[id+16];
		sh_x2[id]+=sh_x2[id+16];
		sh_x[id]*=one_over_len;
		sh_x2[id]*=one_over_len;	
		std[0]=sqrt(sh_x2[id]-sh_x[id]*sh_x[id]);
	}	

}
//////////////////////////
// Normalize each y_i
///
/// H x L_padding
/// threadIdx.x is L
/// blockIdx.x is H
///////////////////////////////////
__global__ void normalize_lowbit(double * signal,const int len,const int padding_len)
{
	const int tx=threadIdx.x;
	const int bx=blockIdx.x*padding_len;
	
	
	double std =CONST_STD[blockIdx.x];
	std = std>EPS ? 1.0/std : 1.0;
	const double *signal_end=signal+bx+len;
	signal += bx+tx;
	
	
	do
	{
		signal[0]*=std;
		signal+=blockDim.x;
	}while(signal<signal_end);
	
	
}

//////////////////////////
/// H x L_padding
/// threadIdx.x for L_padding
/// blockIdx.x for L_padding
/// blockIdx.y for H
///////////////////////////////////
__global__ void normalize(
							double * signal,        //shifted
							const int len,			// current dim len
							const int stride,		// stride between to element
							const int x_len // low bit len	
						)
{
	const int ix=blockIdx.x*blockDim.x+threadIdx.x;
	const int by=blockIdx.y*stride;
	if(ix<x_len)
	{
		double std =CONST_STD[ix];
		std = std>EPS ? 1.0/std : 1.0;
		signal += by+ix;
		signal[0]*=std;
	}
	
}
//////////////////////////////////////////////
// Exapand the signal by adding them with different noise (realizations)
// Note that this is run within a for loop (b=0;b<y_batch_times;b++))
// The input *signal is shifted due to batch;
///
///  H x L_padding  ->  dH x L x N (Ensemble)
///  Input is shifted by high bit 
///  tx for N
///  shared, using tx for L_padding
///  blockIdx.x for L  / EMD_EXPAND_SHARED_SIZE ( =32)
///  blockIdx.y for dH
// The output is batch_len* max_dim_len* 2*ne
//////////////////////////////////////////////
__global__ void ensemble_expansion_lowbit(double *output, 
										  const double *signal, ///shifted signal
										  const double* noise,   
										  const int H_stride,  //L_padding
										  const int L_len)
{
	const int tx= threadIdx.x;
	const int bx= blockIdx.x*EMD_EXPAND_SHARED_SIZE;  //for L 
	const int by= blockIdx.y;			  //for dH
	
	__shared__ double sh_x[EMD_EXPAND_SHARED_SIZE];
	
	//blockDim.x is len of noise
	signal += by*H_stride + bx + tx;           //signal data
	noise  += bx*blockDim.x + tx;           //noise data
	output += (by*L_len+bx)*blockDim.x*2 +tx;

	int last_len=L_len-bx;
	bool not_last= (last_len>=EMD_EXPAND_SHARED_SIZE);
	if(tx<EMD_EXPAND_SHARED_SIZE)
	{
		sh_x[tx]=bx+tx<L_len? signal[0] : 0.0;
		//sh_x[tx]=bx+tx<L_len?  by*100*+bx + tx : 0.0;
	}
	__syncthreads();
	
	double local_noise=noise[0]; noise+=blockDim.x;
	double prefetech;		
	int i;
	if(not_last)
	{
	
		for(i=0;i< (EMD_EXPAND_SHARED_SIZE-1);i++)
		{
			prefetech=noise[0]; noise+=blockDim.x;	//load 1
			output[0]=sh_x[i]+local_noise;   output+=blockDim.x;
			output[0]=sh_x[i]-local_noise;   output+=blockDim.x;
		//	output[0]=sh_x[i];   output+=blockDim.x;
			//output[0]=sh_x[i];   output+=blockDim.x;
		//	output[0]=i;   output+=blockDim.x;
		//	output[0]=i;   output+=blockDim.x;
			local_noise=prefetech;
		}
	}
	else
	{
		for(i=0;i< (last_len-1);i++)
		{
			prefetech=noise[0]; noise+=blockDim.x;	//load 1
			output[0]=sh_x[i]+local_noise;   output+=blockDim.x;
			output[0]=sh_x[i]-local_noise;   output+=blockDim.x;
	//		output[0]=sh_x[i];   output+=blockDim.x;
		//	output[0]=sh_x[i];   output+=blockDim.x;
		//	output[0]=i;   output+=blockDim.x;
		//	output[0]=i;   output+=blockDim.x;
			local_noise=prefetech;
		}
	}

	//last one 
	output[0]=sh_x[i-1]+local_noise;   output+=blockDim.x;
	output[0]=sh_x[i-1]-local_noise; 
	//output[0]=sh_x[i];   output+=blockDim.x;
	//output[0]=sh_x[i]; 
//	output[0]=i;   output+=blockDim.x;
//	output[0]=i;   
	
}


//////////////////////////////////////////////
///  H x L_padding  ->  H x dL x N
///  Input is shifted by high bit
///  tx for N
///  shared, using tx for L_padding
///  blockIdx.x for L
///  blockIdx.y for dH
//////////////////////////////////////////////
__global__ void ensemble_expansion(
	double *output, 
	const double *signal,  //shifted signal
	const double* noise, 
	const int H_stride,  //L_padding
	const int L_len
)
{
	const int tx= threadIdx.x;
	const int bx= blockIdx.x*EMD_EXPAND_SHARED_SIZE;  //for L 
	const int by= blockIdx.y;			  //for dH
	
	__shared__ double sh_x[EMD_EXPAND_SHARED_SIZE];
	
	//blockDim.x is len of noise
	signal += by*H_stride + bx + tx;           //signal data
	noise  += by*blockDim.x + tx;           //noise data
	output += (by*L_len+bx)*blockDim.x*2 +tx;

	int last_len=L_len-bx;
	bool not_last= (last_len>=EMD_EXPAND_SHARED_SIZE);
	if(tx<EMD_EXPAND_SHARED_SIZE)
	{
		sh_x[tx]=bx+tx<L_len? signal[0] : 0.0;
	}
	__syncthreads();
	
	double local_noise=noise[0]; 
	
	int i;
	if(not_last)
	{
	
		for(i=0;i< (EMD_EXPAND_SHARED_SIZE-1);i++)
		{
			output[0]=sh_x[i]+local_noise;   output+=blockDim.x;
			output[0]=sh_x[i]-local_noise;   output+=blockDim.x;
		}
	}
	else
	{
		for(i=0;i< (last_len-1);i++)
		{
			output[0]=sh_x[i]+local_noise;   output+=blockDim.x;
			output[0]=sh_x[i]-local_noise;   output+=blockDim.x;
		}
	}

	//last one 
	output[0]=sh_x[i-1]+local_noise;   output+=blockDim.x;
	output[0]=sh_x[i-1]-local_noise; 
	
}

//////////////////////////////////////////////////
// Find the max and min
///  if dH x L x N
///	 threadIdx.x is for N (only partial)
///  blockIdx.x is for N (segment #)
///  blockIdx.y is for dH  (stride #)
///  len is L
///  high_stride is L*N = 2*ne*x_len
///  jump is N = 2*ne
////////////////////////////////////////////////////

__global__ void extreme(
	double *max_value, double *min_value,
	int *max_array_index, int *min_array_index,
	int *max_len,int *min_len,
	const double * input, const int len,
	const int high_stride, //max_dim_len (x_len)
	const int jump) //1
{
	
	//global index
	///const int tx=threadIdx.x;
	///const int ix=blockIdx.y*high_stride+blockIdx.x*blockDim.x+tx;
	//const int ix=threadIdx.x*high_stride + blockIdx.x*high_stride*threadN;
	const int tx=threadIdx.x + blockIdx.x*blockDim.x;
	const int ix = tx*high_stride;
	//const int jump = 1;
	//extreme points
	int max_p=1;
	int min_p=1;
	
	
	//address
	max_array_index+=ix;
	min_array_index+=ix;
	max_value+=ix;
	min_value+=ix;
	input+=ix;


	double first=input[0];   	input+=jump;
	double second=input[0];  	input+=jump;
	double third=input[0];   	input+=jump;
	max_array_index[0]=0;	max_array_index+=jump;
	min_array_index[0]=0;	min_array_index+=jump;	
	max_value[0]=first; max_value+=jump;
	min_value[0]=first; min_value+=jump;
	

	
	
	for(int i=3;i<len;i++)
	{
		double prefetch=input[0];  	input+=jump;   //4
		
		// for debugging
		//max_array_index[0] = i-2; 		 max_array_index+=jump;
		//max_value[0]=second;		 max_value+=jump;

		//max
		if(second>=first)
		{
			if(second>=third)
			{
				max_array_index[0]=i-2;		 max_array_index+=jump;
				max_value[0]=second;		 max_value+=jump;
				max_p++;

			}
		}
		
		//min
		if(second<=first)
		{
			if(second<=third)
			{
				min_array_index[0]=i-2;		 min_array_index+=jump;
				min_value[0]=second;		 min_value+=jump;
				min_p++;

			}
		}
		
		first=second;
		second=third;
		third=prefetch;
	}
	
	
	
	//end -1
	//max
	if(second>=first)
	{
		if(second>=third)
		{
			//write out
			max_array_index[0]=len-2;	 max_array_index+=jump;
			max_value[0]=second;		 max_value+=jump;
			max_p++;

		}
	}
	//min
	if(second<=first)
	{
		if(second<=third)
		{
			min_array_index[0]=len-2;		min_array_index+=jump;
			min_value[0]=second;		 	min_value+=jump;
			min_p++;
		}
	}
	

	//last (third)
	//write out
	max_array_index[0]=len-1;	
	max_value[0]=third;		 
	max_p++;
	
	
	min_array_index[0]=len-1;
	min_value[0]=third;	
	min_p++;
	
	
	//*int len_ix=blockIdx.y*gridDim.x*blockDim.x+blockIdx.x*blockDim.x+threadIdx.x;
	int len_ix=tx;
	max_len[len_ix]=max_p;
	min_len[len_ix]=min_p;
	
	
}

__global__ void zerocross(
	double *zero_value,	double *zero_index, int *zero_len,
	const double * input, const int len,
	const int high_stride, //max_dim_len (x_len)
	const int jump) //1
{
	//jump = 1;
	//global index
	//const int tx=threadIdx.x;
	//const int ix=blockIdx.y*high_stride+blockIdx.x*blockDim.x+tx;
	//const int ix=tx*high_stride;

	const int tx=threadIdx.x + blockIdx.x*blockDim.x;
	const int ix = tx*high_stride;
//     const int tx=threadIdx.x;
// 	const int ix = tx*high_stride+blockIdx.x*blockDim.x*high_stride;
	//zero points
	int zero_p=0;
	
	//address
	zero_index+=ix;
	zero_value+=ix;
	input+=ix;

	double first=input[0];   	input+=jump;
	double second=input[0];  	input+=jump;

	//zero_index[0]=0;	zero_index+=jump;
	//zero_value[0]=2;    zero_value+=jump;

	
	for(int i=2;i<len;i++)
	{
		double prefetch=input[0];  	input+=jump;   //4
        if(((first> 0) && (second < 0)) || ((first < 0) && (second> 0)))
        {
            //Estimate where zero crossing actually occurs
            double slope = second-first; //(delta x = 1)
            double b = second - (slope *(i));
            zero_index[0] = -b/slope-1;   
            zero_index+=jump;
            zero_value[0] = 0;  zero_value+=jump;
            zero_p++;
        }
        else if((first == 0) || (second == 0))
        {
            zero_index[0] = i;  zero_index+=jump;
            zero_value[0] = 0;  zero_value+=jump;
            zero_p++;
        }
		first=second;
		second=prefetch;
	//*int len_ix=blockIdx.y*gridDim.x*blockDim.x+blockIdx.x*blockDim.x+threadIdx.x;
	int len_ix=tx;
	zero_len[len_ix]=zero_p;
    } 
/*
    if(((first> 0) && (second < 0)) || ((first < 0) && (second> 0)))
    {
        //Estimate where zero crossing actually occurs
        double slope = second-first; //(delta x = 1)
        double b = second - (slope *(i));
        zero_index[0] = -b/slope-1;   
        zero_index+=jump;
        zero_value[0] = 0;  zero_value+=jump;
        zero_p++;
    }
    else if((first == 0) || (second == 0))
    {
        zero_index[0] = i;  zero_index+=jump;
        zero_value[0] = 0;  zero_value+=jump;
        zero_p++;
     }*/
}



__global__ void critical(
	double *value, double *index, int *o_len,
	const double * input, const int len,
	const int high_stride, //max_dim_len (x_len)
	const int jump) //1
{
	
	//global index
	///const int tx=threadIdx.x;
	///const int ix=blockIdx.y*high_stride+blockIdx.x*blockDim.x+tx;
	//const int ix=threadIdx.x*high_stride + blockIdx.x*high_stride*threadN;
	const int tx=threadIdx.x + blockIdx.x*blockDim.x;
	const int ix = tx*high_stride;
	//const int jump = 1;
	//extreme points
	int p=0;
	//address
	index+=ix*2;
	value+=ix*2;
	input+=ix;

	double first=input[0];   	input+=jump;
	double second=input[0];  	input+=jump;
	double third=input[0];   	input+=jump;
// 	index[0]=0;	index+=jump;
// 	value[0]=first; value+=jump;
	
	for(int i=3;i<len;i++)
	{
		double prefetch=input[0];  	input+=jump;   //4
		
		// for debugging
		//max_array_index[0] = i-2; 		 max_array_index+=jump;
		//max_value[0]=second;		 max_value+=jump;

		//max
		if(second>first)
		{
            
			if(second>=third)
			{
				index[0]=i-2;		 index+=jump;
				value[0]=second;		 value+=jump;
				p++;
			}
		}
		
		//min
		if(second<first)
		{
			if(second<=third)
			{
				index[0]=i-2;		 index+=jump;
				value[0]=second;		value+=jump;
				p++;
			}
		}

        //zero crossing
		if(((first> 0) && (second < 0)) || ((first < 0) && (second> 0)))
        {
            //Estimate where zero crossing actually occurs
            double slope = second-first; //(delta x = 1)
            double b = second - (slope *(i));
            index[0] = -b/slope-2;   
            index+=jump;
            value[0] = 0;  value+=jump;
            p++;
        }
        else if((first == 0) || (second == 0))
        {
            index[0] = i-2;  index+=jump;
            value[0] = 0;  value+=jump;
            p++;
        }
		first=second;
		second=third;
		third=prefetch;
	}
	
	//end -1
    //zero crossing
    if(((first> 0) && (second < 0)) || ((first < 0) && (second> 0)))
    {
        //Estimate where zero crossing actually occurs
        double slope = second-first; //(delta x = 1)
        double b = second - (slope *(len));
        index[0] = -b/slope-2;      index+=jump;
        value[0] = 0;       value+=jump;
        p++;
    }
    else if((second == 0) || (third == 0))
    {
        index[0] = len-2;   index+=jump;
        value[0] = 0;       value+=jump;
        p++;
    }
	//max
	if(second>=first)
	{
		if(second>=third)
		{
			//write out
			index[0]=len-2;     index+=jump;
			value[0]=second;	value+=jump;
			p++;

		}
	}
	//min
	if(second<=first)
	{
		if(second<=third)
		{
			index[0]=len-2;		index+=jump;
			value[0]=second;	value+=jump;
			p++;
		}
	}
	

	//last (third)
    //zero crossing
    if(((second> 0) && (third < 0)) || ((second < 0) && (third> 0)))
    {
        //Estimate where zero crossing actually occurs
        double slope = third-second; //(delta x = 1)
        double b = third - (slope *(len));
        index[0] = -b/slope-1;      index+=jump;
        value[0] = 0;       value+=jump;
        p++;
    }
    else if((second == 0) || (third == 0))
    {
        index[0] = len-2;   index+=jump;
        value[0] = 0;       value+=jump;
        p++;
    }
	//write out
// 	index[0]=len-1;	
// 	value[0]=third;		 
// 	p++;

	//*int len_ix=blockIdx.y*gridDim.x*blockDim.x+blockIdx.x*blockDim.x+threadIdx.x;
	int len_ix=tx;
	o_len[len_ix]=p;
}

///////////////////////////////////////////////////////////////////////
// extreme_amp: Extreme points on both local max and min
// max_array_index:  positions of extrema (both local_max and local_min)
// max_value: 		 absolute values of the extrema
// by Hui-Wen, modified from extreme
///////////////////////////////////////////////////////////////////////
__global__ void extreme_amp(
	double *max_value, double *min_value,
	int *max_array_index, int *min_array_index,
	int *max_len,int *min_len,
	const double * input, const int len,
	const int high_stride, //max_dim_len (x_len)
	const int jump) //1
{
	
	//global index
	///const int tx=threadIdx.x;
	///const int ix=blockIdx.y*high_stride+blockIdx.x*blockDim.x+tx;
	//const int ix=threadIdx.x*high_stride + blockIdx.x*high_stride*threadN;
	const int tx=threadIdx.x + blockIdx.x*blockDim.x;
	const int ix = tx*high_stride;
	//const int jump = 1;
	//extreme points
	int max_p=1;
	int min_p=1;
	
	
	//address
	max_array_index+=ix;
	min_array_index+=ix;
	max_value+=ix;
	min_value+=ix;
	input+=ix;


	double first=abs(input[0]);   	input+=jump;
	double second=abs(input[0]);  	input+=jump;
	double third=abs(input[0]);   	input+=jump;
	max_array_index[0]=0;	max_array_index+=jump;
	min_array_index[0]=0;	min_array_index+=jump;	
	max_value[0]=abs(first); max_value+=jump;
	min_value[0]=first; min_value+=jump;
	

	
	
	for(int i=3;i<len;i++)
	{
		double prefetch=abs(input[0]);  	input+=jump;   //4
		
		// for debugging
		//max_array_index[0] = i-2; 		 max_array_index+=jump;
		//max_value[0]=second;		 max_value+=jump;

		//max
		if(second>=first)
		{
			if(second>=third)
			{
				max_array_index[0]=i-2;		 max_array_index+=jump;
				max_value[0]=second;		 max_value+=jump;
				max_p++;
			}
		}
		
		//min
		if(second<=first)
		{
			if(second<=third)
			{
				min_array_index[0]=i-2;		 min_array_index+=jump;
				min_value[0]=second;		 min_value+=jump;
				min_p++;
			}
		}
		
		first=second;
		second=third;
		third=prefetch;
	}
	
	
	
	//end -1
	//max
	if(second>=first)
	{
		if(second>=third)
		{
			//write out
			max_array_index[0]=len-2;	 max_array_index+=jump;
			max_value[0]=second;		 max_value+=jump;
			max_p++;
		}
	}
	//min
	if(second<=first)
	{
		if(second<=third)
		{
			min_array_index[0]=len-2;		min_array_index+=jump;
			min_value[0]=second;		 	min_value+=jump;
			min_p++;
		}
	}
	

	//last (third)
	//write out
	max_array_index[0]=len-1;	
	max_value[0]=third;		 
	max_p++;
	
	
	min_array_index[0]=len-1;
	min_value[0]=third;	
	min_p++;
	
	
	//*int len_ix=blockIdx.y*gridDim.x*blockDim.x+blockIdx.x*blockDim.x+threadIdx.x;
	int len_ix=tx;
	max_len[len_ix]=max_p;
	min_len[len_ix]=min_p;
	
	
}

//////////////////////////////////////////////////
// Find cubic spline
///  if dH x L x N
///	 threadIdx.x is for N (only partial)
///  blockIdx.x is for N (segment #)
///  blockIdx.y is for dH  (stride #)
///  len is L
///  high_stride is L*N
///  jump is N
// Note that in all the ensemble (N) is segmented;
// blockIDx.x for each segment
// threadIdx.x for each x_i in each segment
///
///  if H x dL x N
///	 threadIdx.x is for N (only partial)
///  blockIdx.x is for N (segment #)
///  blockIdx.y is for dL  (stride #)
///  len is H
///  high_stride is N
///  jump is dL x N
////////////////////////////////////////////////////

__global__ void forward(
	double *max_y2, double * min_y2,
	double *max_u, double *min_u,
	double *max_value, double *min_value,
	int *max_array_index, int *min_array_index,
	int *max_len,int *min_len,
	const int high_stride,
	const int jump)
{
	
	//global index
	//const int tx=threadIdx.x;
	//const int ix=blockIdx.y*high_stride+blockIdx.x*blockDim.x+tx;
	
	//int len_ix=blockIdx.y*gridDim.x*blockDim.x+blockIdx.x*blockDim.x+threadIdx.x;
	const int tx=threadIdx.x + blockIdx.x*blockDim.x;
    const int ix = tx*high_stride;
    int len_ix = tx;
	
	max_array_index+=ix;
	min_array_index+=ix;
	max_value+=ix;
	min_value+=ix;
	max_y2+=ix;
	min_y2+=ix;
	max_u+=ix;
	min_u+=ix;
	
	max_y2[0]=0.0;  max_y2+=jump;
	min_y2[0]=0.0;  min_y2+=jump;
	max_u[0]=0.0;   max_u+=jump;
	min_u[0]=0.0;   min_u+=jump;
	
	
	
	double v_0;
	double v_1;
	double v_2;
	int i_0;
	int i_1;
	int i_2;
	double y2_0;
	double u_0;
	
	double sig;
	double p;
	
	y2_0=0.0;
	u_0=0.0;
	int n=max_len[len_ix];
	if(n>=4)
	{
		
		v_0=max_value[0];
	    v_1=max_value[jump];
	    v_2=max_value[2*jump];
		
		i_0=max_array_index[0]; max_array_index+=jump;
		i_1=max_array_index[0];	max_array_index+=jump;
		i_2=max_array_index[0];	max_array_index+=jump;
		
		//update 0
		sig=(v_1-v_2);
		sig/=(i_1-i_2);
		sig*=(i_0-i_1);
		sig+=v_1;
		if(v_0<sig)
		{
			max_value[0]=sig;
			v_0=sig;
		}
		max_value+=3*jump;
		

		for(int i=3;i<n;i++)
		{
			double prefetech_v=max_value[0];	max_value+=jump;
			int prefetech_i=max_array_index[0];	max_array_index+=jump;
			
			//update y2 and u
			sig=(i_1-i_0);
			sig/=(i_2-i_0);
			p=sig*y2_0+2.0;
			y2_0=(sig-1.0)/p;
			u_0=(((v_2-v_1)/(i_2-i_1)-(v_1-v_0)/(i_1-i_0))*6.0/(i_2-i_0)-sig*u_0)/p;
		
			//write out
			max_y2[0]=y2_0; 	max_y2+=jump;
			max_u[0]=u_0;		max_u+=jump;
			
			v_0=v_1;
			v_1=v_2;
			v_2=prefetech_v;
			
			i_0=i_1;
			i_1=i_2;
			i_2=prefetech_i;
		
		}
		
		//last one
		
		//update last one
		sig=(v_1-v_0);
		sig/=(i_1-i_0);
		sig*=(i_2-i_1);
		sig+=v_1;
		if(v_2<sig)
		{
			max_value-=jump;
			max_value[0]=sig;
			v_2=sig;
		}
		
		//update y2 and u
		sig=(i_1-i_0);
		sig/=(i_2-i_0);
		p=sig*y2_0+2.0;
		y2_0=(sig-1.0)/p;
		u_0=(((v_2-v_1)/(i_2-i_1)-(v_1-v_0)/(i_1-i_0))*6.0/(i_2-i_0)-sig*u_0)/p;
	
		//write out
		max_y2[0]=y2_0; 	max_y2+=jump;
		max_u[0]=u_0;		max_u+=jump;
		
		
		
	}
	else
	{
		if(n==3)
		{
			v_0=max_value[0];	max_value+=jump;
			v_1=max_value[0];	max_value+=jump;
			v_2=max_value[0];	max_value+=jump;
		
			i_0=max_array_index[0];   max_array_index+=jump;
			i_1=max_array_index[0];   max_array_index+=jump;
			i_2=max_array_index[0];	  max_array_index+=jump;
			
			//update y2 and u
			sig=(i_1-i_0);
			sig/=(i_2-i_0);
			p=sig*y2_0+2.0;
			y2_0=(sig-1.0)/p;
			u_0=(((v_2-v_1)/(i_2-i_1)-(v_1-v_0)/(i_1-i_0))*6.0/(i_2-i_0)-sig*u_0)/p;
		
			//write out
			max_y2[0]=y2_0; 	max_y2+=jump;
			max_u[0]=u_0;		max_u+=jump;
		}
		
	}
	max_y2[0]=0.0;
	
	y2_0=0.0;
	u_0=0.0;
	n=min_len[len_ix];
	if(n>=4)
	{
		
		v_0=min_value[0];
	    v_1=min_value[jump];
	    v_2=min_value[2*jump];
		
		i_0=min_array_index[0];   min_array_index+=jump;
		i_1=min_array_index[0];	  min_array_index+=jump;
		i_2=min_array_index[0];	  min_array_index+=jump;
		
		//update 0
		sig=(v_1-v_2);
		sig/=(i_1-i_2);
		sig*=(i_0-i_1);
		sig+=v_1;
		if(v_0>sig)
		{
			min_value[0]=sig;
			v_0=sig;
		}
		min_value+=3*jump;
		

		for(int i=3;i<n;i++)
		{
			double prefetech_v=min_value[0];	min_value+=jump;
			int prefetech_i=min_array_index[0];	min_array_index+=jump;
			
			//update y2 and u
			sig=(i_1-i_0);
			sig/=(i_2-i_0);
			p=sig*y2_0+2.0;
			y2_0=(sig-1.0)/p;
			u_0=(((v_2-v_1)/(i_2-i_1)-(v_1-v_0)/(i_1-i_0))*6.0/(i_2-i_0)-sig*u_0)/p;
		
			//write out
			min_y2[0]=y2_0; 	min_y2+=jump;
			min_u[0]=u_0;		min_u+=jump;
			
			v_0=v_1;
			v_1=v_2;
			v_2=prefetech_v;
			
			i_0=i_1;
			i_1=i_2;
			i_2=prefetech_i;
		
		}
		
		//last one
		
		//update last one
		sig=(v_1-v_0);
		sig/=(i_1-i_0);
		sig*=(i_2-i_1);
		sig+=v_1;
		if(v_2>sig)
		{
			min_value-=jump;
			min_value[0]=sig;
			v_2=sig;
		}
		
		//update y2 and u
		sig=(i_1-i_0);
		sig/=(i_2-i_0);
		p=sig*y2_0+2.0;
		y2_0=(sig-1.0)/p;
		u_0=(((v_2-v_1)/(i_2-i_1)-(v_1-v_0)/(i_1-i_0))*6.0/(i_2-i_0)-sig*u_0)/p;
	
		//write out
		min_y2[0]=y2_0; 	min_y2+=jump;
		min_u[0]=u_0;		min_u+=jump;
		
		
		
	}
	else
	{
		if(n==3)
		{
			v_0=min_value[0];	min_value+=jump;
			v_1=min_value[0];	min_value+=jump;
			v_2=min_value[0];	min_value+=jump;
		
			i_0=min_array_index[0];   min_array_index+=jump;
			i_1=min_array_index[0];   min_array_index+=jump;
			i_2=min_array_index[0];	  min_array_index+=jump;
			//update y2 and u
			sig=(i_1-i_0);
			sig/=(i_2-i_0);
			p=sig*y2_0+2.0;
			y2_0=(sig-1.0)/p;
			u_0=(((v_2-v_1)/(i_2-i_1)-(v_1-v_0)/(i_1-i_0))*6.0/(i_2-i_0)-sig*u_0)/p;
		
			//write out
			min_y2[0]=y2_0; 	min_y2+=jump;
			min_u[0]=u_0;		min_u+=jump;
		}
		
	}
	min_y2[0]=0.0;
		
}



__global__ void extreme_forward(double *max_y2, double * min_y2,
								double *max_u, double *min_u,
								double *max_value, double *min_value,
							    int *max_array_index, int *min_array_index,
							    int *max_len,int *min_len,
							    const double * input, const int len,
								const int high_stride,
								const int jump)
{
	
	//global index
	const int tx=threadIdx.x;
	const int ix=blockIdx.y*high_stride+blockIdx.x*blockDim.x+tx;
	
	
	
	//extreme points
	int max_p=1;
	int min_p=1;
	
	bool max_3=true;
	bool min_3=true;
	
	//address
	max_array_index+=ix;
	min_array_index+=ix;
	max_value+=ix;
	min_value+=ix;
	input+=ix;
	max_y2+=ix;
	min_y2+=ix;
	max_u+=ix;
	min_u+=ix;

	double first=input[0];   	input+=jump;
	double second=input[0];  	input+=jump;
	double third=input[0];   	input+=jump;
	max_array_index[0]=0;	max_array_index+=jump;
	min_array_index[0]=0;	min_array_index+=jump;	
	double *max_value_0=max_value;
	double *min_value_0=min_value;
	max_value+=jump;
	min_value+=jump;
	
	max_y2[0]=0.0;  max_y2+=jump;
	min_y2[0]=0.0;  min_y2+=jump;
	max_u[0]=0.0;   max_u+=jump;
	min_u[0]=0.0;   min_u+=jump;

	__shared__ double max_v[3][THREAD_MAIN];
	__shared__ double min_v[3][THREAD_MAIN];
	__shared__ int max_i[3][THREAD_MAIN];
	__shared__ int min_i[3][THREAD_MAIN];

	
	
	max_v[0][tx]=first;  
	min_v[0][tx]=first;
	max_v[1][tx]=first;  
	min_v[1][tx]=first;
	max_v[2][tx]=first;  
	min_v[2][tx]=first;
	
	max_i[0][tx]=0;
	min_i[0][tx]=0;
	max_i[1][tx]=1;
	min_i[1][tx]=1;
	max_i[2][tx]=2;
	min_i[2][tx]=2;
	double max_y2_0=0.0;
	double max_u_0=0.0;
	double min_y2_0=0.0;
	double min_u_0=0.0;
	
	double sig;
	double p;

	
	
	for(int i=3;i<len;i++)
	{
		double prefetch=input[0];  	input+=jump;   //4
		
		//max
		if(second>=first)
		{
			if(second>=third)
			{
				max_array_index[0]=i-2;		 max_array_index+=jump;
				max_value[0]=second;		 max_value+=jump;
				//max_value[0]=i-3.0;		 max_value+=jump;
				max_p++;
				//first 3?
				if(max_3)
				{
					
					if(max_p==2)
					{
						max_v[1][tx]=second;
						max_i[1][tx]=i-2;
					}
					else if(max_p==3)
					{
						max_v[2][tx]=second;
						max_i[2][tx]=i-2;
					}
					else   // fourth
					{ 
						max_3=false;
						
						//update max_v[0][tx]
						sig=(max_v[1][tx]-max_v[2][tx]);
						sig/=(max_i[1][tx]-max_i[2][tx]);
						sig*=(max_i[0][tx]-max_i[1][tx]);
						sig+=max_v[1][tx];
						max_v[0][tx]=max_v[0][tx]<sig?sig:max_v[0][tx];
						max_value_0[0]=max_v[0][tx];
						
						//calculate u and y2
						sig=(max_i[1][tx]-max_i[0][tx]);
						sig/=(max_i[2][tx]-max_i[0][tx]);
						p=sig*max_y2_0+2.0;
						max_y2_0=(sig-1.0)/p;
						max_u_0=(((max_v[2][tx]-max_v[1][tx])/(max_i[2][tx]-max_i[1][tx])-(max_v[1][tx]-max_v[0][tx])/(max_i[1][tx]-max_i[0][tx]))*6.0/(max_i[2][tx]-max_i[0][tx])-sig*max_u_0)/p;
						
						//write out
						max_y2[0]=max_y2_0; 	max_y2+=jump;
						max_u[0]=max_u_0;		max_u+=jump;
						
						//update data
						max_v[0][tx]=max_v[1][tx];
						max_v[1][tx]=max_v[2][tx];
						max_v[2][tx]=second;
						max_i[0][tx]=max_i[1][tx];
						max_i[1][tx]=max_i[2][tx];
						max_i[2][tx]=i-2;
						
						//calculate new y2 and u
						sig=(max_i[1][tx]-max_i[0][tx]);
						sig/=(max_i[2][tx]-max_i[0][tx]);
						p=sig*max_y2_0+2.0;
						max_y2_0=(sig-1.0)/p;
						max_u_0=(((max_v[2][tx]-max_v[1][tx])/(max_i[2][tx]-max_i[1][tx])-(max_v[1][tx]-max_v[0][tx])/(max_i[1][tx]-max_i[0][tx]))*6.0/(max_i[2][tx]-max_i[0][tx])-sig*max_u_0)/p;
						//write out
						max_y2[0]=max_y2_0; 	max_y2+=jump;
						max_u[0]=max_u_0;		max_u+=jump;

						
					}

				}
				else
				{
				
					//update data
					max_v[0][tx]=max_v[1][tx];
					max_v[1][tx]=max_v[2][tx];
					max_v[2][tx]=second;
					max_i[0][tx]=max_i[1][tx];
					max_i[1][tx]=max_i[2][tx];
					max_i[2][tx]=i-2;
					sig=(max_i[1][tx]-max_i[0][tx]);
					sig/=(max_i[2][tx]-max_i[0][tx]);
					p=sig*max_y2_0+2.0;
					max_y2_0=(sig-1.0)/p;
					max_u_0=(((max_v[2][tx]-max_v[1][tx])/(max_i[2][tx]-max_i[1][tx])-(max_v[1][tx]-max_v[0][tx])/(max_i[1][tx]-max_i[0][tx]))*6.0/(max_i[2][tx]-max_i[0][tx])-sig*max_u_0)/p;
					
					//write out
					max_y2[0]=max_y2_0; 	max_y2+=jump;
					max_u[0]=max_u_0;		max_u+=jump;
				}
			}
		}
		
		//min
		if(second<=first)
		{
			if(second<=third)
			{
				min_array_index[0]=i-2;		 min_array_index+=jump;
				min_value[0]=second;		 min_value+=jump;
				min_p++;
				//first 3?
				if(min_3)
				{
					
					if(min_p==2)
					{
						min_v[1][tx]=second;
						min_i[1][tx]=i-2;
					}
					else if(min_p==3)
					{
						min_v[2][tx]=second;
						min_i[2][tx]=i-2;
	
					}
					else
					{
						min_3=false;
						//update min_v[0][tx]
						sig=(min_v[1][tx]-min_v[2][tx]);
						sig/=(min_i[1][tx]-min_i[2][tx]);
						sig*=(min_i[0][tx]-min_i[1][tx]);
						sig+=min_v[1][tx];
						min_v[0][tx]=min_v[0][tx]>sig?sig:min_v[0][tx];
						min_value_0[0]=min_v[0][tx];
						
						//update y2 and u
						sig=(min_i[1][tx]-min_i[0][tx]);
						sig/=(min_i[2][tx]-min_i[0][tx]);
						p=sig*min_y2_0+2.0;
						min_y2_0=(sig-1.0)/p;
						min_u_0=(((min_v[2][tx]-min_v[1][tx])/(min_i[2][tx]-min_i[1][tx])-(min_v[1][tx]-min_v[0][tx])/(min_i[1][tx]-min_i[0][tx]))*6.0/(min_i[2][tx]-min_i[0][tx])-sig*min_u_0)/p;
						
						//write out
						min_y2[0]=min_y2_0; 	min_y2+=jump;
						min_u[0]=min_u_0;		min_u+=jump;
						
						//update data
						min_v[0][tx]=min_v[1][tx];
						min_v[1][tx]=min_v[2][tx];
						min_v[2][tx]=second;
						min_i[0][tx]=min_i[1][tx];
						min_i[1][tx]=min_i[2][tx];
						min_i[2][tx]=i-2;

						//update new y2 and u
						sig=(min_i[1][tx]-min_i[0][tx]);
						sig/=(min_i[2][tx]-min_i[0][tx]);
						p=sig*min_y2_0+2.0;
						min_y2_0=(sig-1.0)/p;
						min_u_0=(((min_v[2][tx]-min_v[1][tx])/(min_i[2][tx]-min_i[1][tx])-(min_v[1][tx]-min_v[0][tx])/(min_i[1][tx]-min_i[0][tx]))*6.0/(min_i[2][tx]-min_i[0][tx])-sig*min_u_0)/p;
						//write out
						min_y2[0]=min_y2_0; 	min_y2+=jump;
						min_u[0]=min_u_0;		min_u+=jump;
					}
				}
				else
				{					
					//update data
					min_v[0][tx]=min_v[1][tx];
					min_v[1][tx]=min_v[2][tx];
					min_v[2][tx]=second;
					min_i[0][tx]=min_i[1][tx];
					min_i[1][tx]=min_i[2][tx];
					min_i[2][tx]=i-2;
					sig=(min_i[1][tx]-min_i[0][tx]);
					sig/=(min_i[2][tx]-min_i[0][tx]);
					p=sig*min_y2_0+2.0;
					min_y2_0=(sig-1.0)/p;
					min_u_0=(((min_v[2][tx]-min_v[1][tx])/(min_i[2][tx]-min_i[1][tx])-(min_v[1][tx]-min_v[0][tx])/(min_i[1][tx]-min_i[0][tx]))*6.0/(min_i[2][tx]-min_i[0][tx])-sig*min_u_0)/p;
					
					//write out
					min_y2[0]=min_y2_0; 	min_y2+=jump;
					min_u[0]=min_u_0;		min_u+=jump;
				}
			}
		}
		
		first=second;
		second=third;
		third=prefetch;
	}
	
	
	
	//end -1
	//max
	if(second>=first)
	{
		if(second>=third)
		{
			//write out
			max_array_index[0]=len-2;	 max_array_index+=jump;
			max_value[0]=second;		 max_value+=jump;
			max_p++;
			if(max_3)
			{
				
				if(max_p==2)
				{
					max_v[1][tx]=second;
					max_i[1][tx]=len-2;
				}
				else if(max_p==3)
				{
					max_v[2][tx]=second;
					max_i[2][tx]=len-2;
				}
				else   // fourth
				{ 
					max_3=false;
					
					//update max_v[0][tx]
					sig=(max_v[1][tx]-max_v[2][tx]);
					sig/=(max_i[1][tx]-max_i[2][tx]);
					sig*=(max_i[0][tx]-max_i[1][tx]);
					sig+=max_v[1][tx];
					max_v[0][tx]=max_v[0][tx]<sig?sig:max_v[0][tx];
					max_value_0[0]=max_v[0][tx];
					
					//calculate u and y2
					sig=(max_i[1][tx]-max_i[0][tx]);
					sig/=(max_i[2][tx]-max_i[0][tx]);
					p=sig*max_y2_0+2.0;
					max_y2_0=(sig-1.0)/p;
					max_u_0=(((max_v[2][tx]-max_v[1][tx])/(max_i[2][tx]-max_i[1][tx])-(max_v[1][tx]-max_v[0][tx])/(max_i[1][tx]-max_i[0][tx]))*6.0/(max_i[2][tx]-max_i[0][tx])-sig*max_u_0)/p;
					
					//write out
					max_y2[0]=max_y2_0; 	max_y2+=jump;
					max_u[0]=max_u_0;		max_u+=jump;
					
					//update data
					max_v[0][tx]=max_v[1][tx];
					max_v[1][tx]=max_v[2][tx];
					max_v[2][tx]=second;
					max_i[0][tx]=max_i[1][tx];
					max_i[1][tx]=max_i[2][tx];
					max_i[2][tx]=len-2;
					
					//calculate new y2 and u
					sig=(max_i[1][tx]-max_i[0][tx]);
					sig/=(max_i[2][tx]-max_i[0][tx]);
					p=sig*max_y2_0+2.0;
					max_y2_0=(sig-1.0)/p;
					max_u_0=(((max_v[2][tx]-max_v[1][tx])/(max_i[2][tx]-max_i[1][tx])-(max_v[1][tx]-max_v[0][tx])/(max_i[1][tx]-max_i[0][tx]))*6.0/(max_i[2][tx]-max_i[0][tx])-sig*max_u_0)/p;
					//write out
					max_y2[0]=max_y2_0; 	max_y2+=jump;
					max_u[0]=max_u_0;		max_u+=jump;

					
				}

			}
			else
			{
			
				//update data
				max_v[0][tx]=max_v[1][tx];
				max_v[1][tx]=max_v[2][tx];
				max_v[2][tx]=second;
				max_i[0][tx]=max_i[1][tx];
				max_i[1][tx]=max_i[2][tx];
				max_i[2][tx]=len-2;
				sig=(max_i[1][tx]-max_i[0][tx]);
				sig/=(max_i[2][tx]-max_i[0][tx]);
				p=sig*max_y2_0+2.0;
				max_y2_0=(sig-1.0)/p;
				max_u_0=(((max_v[2][tx]-max_v[1][tx])/(max_i[2][tx]-max_i[1][tx])-(max_v[1][tx]-max_v[0][tx])/(max_i[1][tx]-max_i[0][tx]))*6.0/(max_i[2][tx]-max_i[0][tx])-sig*max_u_0)/p;
				
				//write out
				max_y2[0]=max_y2_0; 	max_y2+=jump;
				max_u[0]=max_u_0;		max_u+=jump;
			}
		}
	}
	//min
	if(second<=first)
	{
		if(second<=third)
		{
			min_array_index[0]=len-2;		 min_array_index+=jump;
			min_value[0]=second;		 min_value+=jump;
			min_p++;
			//first 3?
			if(min_3)
			{
				
				if(min_p==2)
				{
					min_v[1][tx]=second;
					min_i[1][tx]=len-2;
				}
				else if(min_p==3)
				{
					min_v[2][tx]=second;
					min_i[2][tx]=len-2;

				}
				else
				{
					min_3=false;
					//update min_v[0][tx]
					sig=(min_v[1][tx]-min_v[2][tx]);
					sig/=(min_i[1][tx]-min_i[2][tx]);
					sig*=(min_i[0][tx]-min_i[1][tx]);
					sig+=min_v[1][tx];
					min_v[0][tx]=min_v[0][tx]>sig?sig:min_v[0][tx];
					min_value_0[0]=min_v[0][tx];
					
					//update y2 and u
					sig=(min_i[1][tx]-min_i[0][tx]);
					sig/=(min_i[2][tx]-min_i[0][tx]);
					p=sig*min_y2_0+2.0;
					min_y2_0=(sig-1.0)/p;
					min_u_0=(((min_v[2][tx]-min_v[1][tx])/(min_i[2][tx]-min_i[1][tx])-(min_v[1][tx]-min_v[0][tx])/(min_i[1][tx]-min_i[0][tx]))*6.0/(min_i[2][tx]-min_i[0][tx])-sig*min_u_0)/p;
					
					//write out
					min_y2[0]=min_y2_0; 	min_y2+=jump;
					min_u[0]=min_u_0;		min_u+=jump;
					
					//update data
					min_v[0][tx]=min_v[1][tx];
					min_v[1][tx]=min_v[2][tx];
					min_v[2][tx]=second;
					min_i[0][tx]=min_i[1][tx];
					min_i[1][tx]=min_i[2][tx];
					min_i[2][tx]=len-2;

					//update new y2 and u
					sig=(min_i[1][tx]-min_i[0][tx]);
					sig/=(min_i[2][tx]-min_i[0][tx]);
					p=sig*min_y2_0+2.0;
					min_y2_0=(sig-1.0)/p;
					min_u_0=(((min_v[2][tx]-min_v[1][tx])/(min_i[2][tx]-min_i[1][tx])-(min_v[1][tx]-min_v[0][tx])/(min_i[1][tx]-min_i[0][tx]))*6.0/(min_i[2][tx]-min_i[0][tx])-sig*min_u_0)/p;
					//write out
					min_y2[0]=min_y2_0; 	min_y2+=jump;
					min_u[0]=min_u_0;		min_u+=jump;
				}
			}
			else
			{					
				//update data
				min_v[0][tx]=min_v[1][tx];
				min_v[1][tx]=min_v[2][tx];
				min_v[2][tx]=second;
				min_i[0][tx]=min_i[1][tx];
				min_i[1][tx]=min_i[2][tx];
				min_i[2][tx]=len-2;
				sig=(min_i[1][tx]-min_i[0][tx]);
				sig/=(min_i[2][tx]-min_i[0][tx]);
				p=sig*min_y2_0+2.0;
				min_y2_0=(sig-1.0)/p;
				min_u_0=(((min_v[2][tx]-min_v[1][tx])/(min_i[2][tx]-min_i[1][tx])-(min_v[1][tx]-min_v[0][tx])/(min_i[1][tx]-min_i[0][tx]))*6.0/(min_i[2][tx]-min_i[0][tx])-sig*min_u_0)/p;
				
				//write out
				min_y2[0]=min_y2_0; 	min_y2+=jump;
				min_u[0]=min_u_0;		min_u+=jump;
			}
		}
	}
	

	//last (third)
	//write out
	max_array_index[0]=len-1;	
	max_p++;
	if(max_3)
	{
		
		if(max_p==2)
		{
			max_value[0]=third;		  
		}
		else if(max_p==3)
		{
			max_value[0]=third;		  
			max_v[2][tx]=third;
			max_i[2][tx]=len-1;
			
			//calculate u and y2
			sig=(max_i[1][tx]-max_i[0][tx]);
			sig/=(max_i[2][tx]-max_i[0][tx]);
			p=sig*max_y2_0+2.0;
			max_y2_0=(sig-1.0)/p;
			max_u_0=(((max_v[2][tx]-max_v[1][tx])/(max_i[2][tx]-max_i[1][tx])-(max_v[1][tx]-max_v[0][tx])/(max_i[1][tx]-max_i[0][tx]))*6.0/(max_i[2][tx]-max_i[0][tx])-sig*max_u_0)/p;
			//write out
			max_y2[0]=max_y2_0; 	max_y2+=jump;
			max_u[0]=max_u_0;	
			
			
		}
		else   // fourth
		{ 

			//update max_v[0][tx]
			sig=(max_v[1][tx]-max_v[2][tx]);
			sig/=(max_i[1][tx]-max_i[2][tx]);
			sig*=(max_i[0][tx]-max_i[1][tx]);
			sig+=max_v[1][tx];
			max_v[0][tx]=max_v[0][tx]<sig?sig:max_v[0][tx];
			max_value_0[0]=max_v[0][tx];
			
			//calculate u and y2
			sig=(max_i[1][tx]-max_i[0][tx]);
			sig/=(max_i[2][tx]-max_i[0][tx]);
			p=sig*max_y2_0+2.0;
			max_y2_0=(sig-1.0)/p;
			max_u_0=(((max_v[2][tx]-max_v[1][tx])/(max_i[2][tx]-max_i[1][tx])-(max_v[1][tx]-max_v[0][tx])/(max_i[1][tx]-max_i[0][tx]))*6.0/(max_i[2][tx]-max_i[0][tx])-sig*max_u_0)/p;
			
			//write out
			max_y2[0]=max_y2_0; 	max_y2+=jump;
			max_u[0]=max_u_0;		max_u+=jump;
			
			//update data
			max_v[0][tx]=max_v[1][tx];
			max_v[1][tx]=max_v[2][tx];
			max_v[2][tx]=third;
			max_i[0][tx]=max_i[1][tx];
			max_i[1][tx]=max_i[2][tx];
			max_i[2][tx]=len-1;
			
			//update max_v[2][tx]
			sig=(max_v[1][tx]-max_v[0][tx]);
			sig/=(max_i[1][tx]-max_i[0][tx]);
			sig*=(max_i[2][tx]-max_i[1][tx]);
			sig+=max_v[1][tx];
			max_v[2][tx]=max_v[2][tx]<sig?sig:max_v[2][tx];
			max_value[0]=max_v[2][tx];
			//max_value[0]=3.0;					
			
			//calculate new y2 and u
			sig=(max_i[1][tx]-max_i[0][tx]);
			sig/=(max_i[2][tx]-max_i[0][tx]);
			p=sig*max_y2_0+2.0;
			max_y2_0=(sig-1.0)/p;
			max_u_0=(((max_v[2][tx]-max_v[1][tx])/(max_i[2][tx]-max_i[1][tx])-(max_v[1][tx]-max_v[0][tx])/(max_i[1][tx]-max_i[0][tx]))*6.0/(max_i[2][tx]-max_i[0][tx])-sig*max_u_0)/p;
			//write out
			max_y2[0]=max_y2_0; 	max_y2+=jump;
			max_u[0]=max_u_0;		

		}

	}
	else
	{
		//update data
		max_v[0][tx]=max_v[1][tx];
		max_v[1][tx]=max_v[2][tx];
		max_v[2][tx]=third;
		max_i[0][tx]=max_i[1][tx];
		max_i[1][tx]=max_i[2][tx];
		max_i[2][tx]=len-1;
		
		//update max_v[2][tx]
		sig=(max_v[1][tx]-max_v[0][tx]);
		sig/=(max_i[1][tx]-max_i[0][tx]);
		sig*=(max_i[2][tx]-max_i[1][tx]);
		sig+=max_v[1][tx];
		max_v[2][tx]=max_v[2][tx]<sig?sig:max_v[2][tx];
		max_value[0]=max_v[2][tx];	
		
		sig=(max_i[1][tx]-max_i[0][tx]);
		sig/=(max_i[2][tx]-max_i[0][tx]);
		p=sig*max_y2_0+2.0;
		max_y2_0=(sig-1.0)/p;
		max_u_0=(((max_v[2][tx]-max_v[1][tx])/(max_i[2][tx]-max_i[1][tx])-(max_v[1][tx]-max_v[0][tx])/(max_i[1][tx]-max_i[0][tx]))*6.0/(max_i[2][tx]-max_i[0][tx])-sig*max_u_0)/p;
		
		//write out
		max_y2[0]=max_y2_0; 	max_y2+=jump;
		max_u[0]=max_u_0;	
	}
	max_y2[0]=0.0; 
	
	
	min_array_index[0]=len-1;		
	min_p++;

	if(min_3)
	{
		
		if(min_p==2)
		{
			min_value[0]=third;	
		}
		else if(min_p==3)
		{
			min_value[0]=third;	
			min_v[2][tx]=third;
			min_i[2][tx]=len-1;
			
			//update y2 and u
			sig=(min_i[1][tx]-min_i[0][tx]);
			sig/=(min_i[2][tx]-min_i[0][tx]);
			p=sig*min_y2_0+2.0;
			min_y2_0=(sig-1.0)/p;
			min_u_0=(((min_v[2][tx]-min_v[1][tx])/(min_i[2][tx]-min_i[1][tx])-(min_v[1][tx]-min_v[0][tx])/(min_i[1][tx]-min_i[0][tx]))*6.0/(min_i[2][tx]-min_i[0][tx])-sig*min_u_0)/p;
			
			//write out
			min_y2[0]=min_y2_0; 	min_y2+=jump;
			min_u[0]=min_u_0;		

		}
		else
		{
		
			//update min_v[0][tx]
			sig=(min_v[1][tx]-min_v[2][tx]);
			sig/=(min_i[1][tx]-min_i[2][tx]);
			sig*=(min_i[0][tx]-min_i[1][tx]);
			sig+=min_v[1][tx];
			min_v[0][tx]=min_v[0][tx]>sig?sig:min_v[0][tx];
			min_value_0[0]=min_v[0][tx];
			
			//update y2 and u
			sig=(min_i[1][tx]-min_i[0][tx]);
			sig/=(min_i[2][tx]-min_i[0][tx]);
			p=sig*min_y2_0+2.0;
			min_y2_0=(sig-1.0)/p;
			min_u_0=(((min_v[2][tx]-min_v[1][tx])/(min_i[2][tx]-min_i[1][tx])-(min_v[1][tx]-min_v[0][tx])/(min_i[1][tx]-min_i[0][tx]))*6.0/(min_i[2][tx]-min_i[0][tx])-sig*min_u_0)/p;
			
			//write out
			min_y2[0]=min_y2_0; 	min_y2+=jump;
			min_u[0]=min_u_0;		min_u+=jump;
			
			//update data
			min_v[0][tx]=min_v[1][tx];
			min_v[1][tx]=min_v[2][tx];
			min_v[2][tx]=third;
			min_i[0][tx]=min_i[1][tx];
			min_i[1][tx]=min_i[2][tx];
			min_i[2][tx]=len-1;
			
			//update max_v[2][tx]
			sig=(min_v[1][tx]-min_v[0][tx]);
			sig/=(min_i[1][tx]-min_i[0][tx]);
			sig*=(min_i[2][tx]-min_i[1][tx]);
			sig+=min_v[1][tx];
			min_v[2][tx]=min_v[2][tx]>sig?sig:min_v[2][tx];
			min_value[0]=min_v[2][tx];	

			//update new y2 and u
			sig=(min_i[1][tx]-min_i[0][tx])/(min_i[2][tx]-min_i[0][tx]);
			p=sig*min_y2_0+2.0;
			min_y2_0=(sig-1.0)/p;
			min_u_0=(((min_v[2][tx]-min_v[1][tx])/(min_i[2][tx]-min_i[1][tx])-(min_v[1][tx]-min_v[0][tx])/(min_i[1][tx]-min_i[0][tx]))*6.0/(min_i[2][tx]-min_i[0][tx])-sig*min_u_0)/p;
			//write out
			min_y2[0]=min_y2_0; 	min_y2+=jump;
			min_u[0]=min_u_0;		
			
		}
	}
	else
	{					
		//update data
		min_v[0][tx]=min_v[1][tx];
		min_v[1][tx]=min_v[2][tx];
		min_v[2][tx]=third;
		min_i[0][tx]=min_i[1][tx];
		min_i[1][tx]=min_i[2][tx];
		min_i[2][tx]=len-1;
		
		//update max_v[2][tx]
		sig=(min_v[1][tx]-min_v[0][tx]);
		sig/=(min_i[1][tx]-min_i[0][tx]);
		sig*=(min_i[2][tx]-min_i[1][tx]);
		sig+=min_v[1][tx];
		min_v[2][tx]=min_v[2][tx]>sig?sig:min_v[2][tx];
		min_value[0]=min_v[2][tx];
		
		
		//update new y2 and u
		sig=(min_i[1][tx]-min_i[0][tx]);
		sig/=(min_i[2][tx]-min_i[0][tx]);
		p=sig*min_y2_0+2.0;
		min_y2_0=(sig-1.0)/p;
		min_u_0=(((min_v[2][tx]-min_v[1][tx])/(min_i[2][tx]-min_i[1][tx])-(min_v[1][tx]-min_v[0][tx])/(min_i[1][tx]-min_i[0][tx]))*6.0/(min_i[2][tx]-min_i[0][tx])-sig*min_u_0)/p;
		
		//write out
		min_y2[0]=min_y2_0; 	min_y2+=jump;
		min_u[0]=min_u_0;		
	}
	min_y2[0]=0.0; 
	
	
	int len_ix=blockIdx.y*gridDim.x*blockDim.x+blockIdx.x*blockDim.x+threadIdx.x;
	max_len[len_ix]=max_p;
	min_len[len_ix]=min_p;
	
	
}

/////////////////////////////////////////////////
///  if dH x L x N
///	 threadIdx.x is for N (only partial)
///  blockIdx.x is for N (segment #)
///  blockIdx.y is for dH  (stride #)
///  len is L
///  high_stride is L*N
///  jump is N
///  if H x dL x N
///	 threadIdx.x is for N (only partial)
///  blockIdx.x is for N (segment #)
///  blockIdx.y is for dL  (stride #)
///  len is H
///  high_stride is N
///  jump is dL x N
///  blockDim.x==blockDim.y
////////////////////////////////////////////////////
__global__ void  backward(double *max_y2,  double *min_y2,
						  double *max_u,   double *min_u,
						  const int *max_len,    const int *min_len, 
						  const int high_stride,
						  const int jump)
{
	const int tx=threadIdx.x;
	const int ty=threadIdx.y;
	const int id=ty*blockDim.x+tx;  //
    const int s_id=tx*blockDim.x+ty;
    // Index for max_y2
	//const int ix=blockIdx.y*high_stride+blockIdx.x*EMD_BACKWARD_SHARED_SIZE+id;
    const int  ix = blockIdx.x*EMD_BACKWARD_SHARED_SIZE*high_stride + id*high_stride;

    // Index for max_len
    //const int len_ix=blockIdx.y*gridDim.x*EMD_BACKWARD_SHARED_SIZE + blockIdx.x*EMD_BACKWARD_SHARED_SIZE+id;
    const int len_ix = blockIdx.x * EMD_BACKWARD_SHARED_SIZE + id;
    __shared__ int sh_bound[EMD_BACKWARD_SHARED_SIZE];
	
	

	max_len+=len_ix;
	int max_bound=max_len[0];
	min_len+=len_ix;
	max_y2+=ix;
	min_y2+=ix;
	max_u+=ix;
	min_u+=ix;
	sh_bound[s_id]=max_bound;
	__syncthreads();

	//find longest bound
	if(id<128) { sh_bound[id]=sh_bound[id]>sh_bound[id+128]?sh_bound[id]:sh_bound[id+128];}	__syncthreads();
	if(id<64) { sh_bound[id]=sh_bound[id]>sh_bound[id+64]?sh_bound[id]:sh_bound[id+64];} 	__syncthreads();
	if(id<32) { sh_bound[id]=sh_bound[id]>sh_bound[id+32]?sh_bound[id]:sh_bound[id+32];}
	if(id<16) { sh_bound[id]=sh_bound[id]>sh_bound[id+16]?sh_bound[id]:sh_bound[id+16];}
	__syncthreads();
	int high_bound=sh_bound[ty];
	//__syncthreads();
	int min_bound=min_len[0];
	//max
	max_y2+=(high_bound-2)*jump;
	max_u+=(high_bound-2)*jump;
	//last one
	double y2_0=0.0;
	double y2_1=0.0;
	int i;
	max_bound-=2;
	for(i=high_bound-2;i>=0;i--)
	{
		if(i<=max_bound)
		{

			y2_0=max_y2[0];
			y2_1=y2_0*y2_1+max_u[0];
			max_y2[0]=y2_1;

		}
		max_y2-=jump;
		max_u-=jump;
	}

	//min
	sh_bound[s_id]=min_bound;
	__syncthreads();
	//find longest bound
	if(id<128) { sh_bound[id]=sh_bound[id]>sh_bound[id+128]?sh_bound[id]:sh_bound[id+128];}	__syncthreads();
	if(id<64) { sh_bound[id]=sh_bound[id]>sh_bound[id+64]?sh_bound[id]:sh_bound[id+64];} 	__syncthreads();
	if(id<32) { sh_bound[id]=sh_bound[id]>sh_bound[id+32]?sh_bound[id]:sh_bound[id+32];}
	if(id<16) { sh_bound[id]=sh_bound[id]>sh_bound[id+16]?sh_bound[id]:sh_bound[id+16];}
	__syncthreads();
	high_bound=sh_bound[ty];

	//min
	min_y2+=(high_bound-2)*jump;
	min_u+=(high_bound-2)*jump;
	//last one
	y2_0=0.0;
	y2_1=0.0;
	min_bound-=2;
	for( i=high_bound-2;i>=0;i--)
	{
		if(i<=min_bound)
		{
			y2_0=min_y2[0];
			y2_1=y2_0*y2_1+min_u[0];
			min_y2[0]=y2_1;
		}
		min_y2-=jump;
		min_u-=jump;
	}
	
	
}




__global__ void spline_interp_0(double *output,
								const double *input,
								const double * max_y2, const double * min_y2,
								const double * max_array_value,const double * min_array_value,
								const int *max_array_index,const int *min_array_index,
								const int len,
								const int high_stride,
								const int jump
								)
{
// 	const int tx=threadIdx.x;
// 	const int ix=blockIdx.y*high_stride+blockIdx.x*blockDim.x+tx;
	const int tx=threadIdx.x;
	const int ix = tx*high_stride+blockIdx.x*blockDim.x*high_stride;

	max_array_value+=ix;
	min_array_value+=ix;
	max_array_index+=ix;
	min_array_index+=ix;
	max_y2+=ix;
	min_y2+=ix;
	output+=ix;
	input+=ix;
	
	__shared__ double one_over_6;
	__shared__ double max_x_diff[THREAD_MAIN];
	__shared__ double min_x_diff[THREAD_MAIN];
	
	if(tx<1)
	{
		one_over_6=1.0/6.0;
	}
	__syncthreads();
	
	//x ids;
	int max_x0=max_array_index[0]; max_array_index+=jump;
	int max_x1=max_array_index[0]; max_array_index+=jump;
	
	int min_x0=min_array_index[0]; min_array_index+=jump;
	int min_x1=min_array_index[0]; min_array_index+=jump;
	
	//y value
	double max_y0=max_array_value[0]; max_array_value+=jump;
	double max_y1=max_array_value[0]; max_array_value+=jump;
	
	double min_y0=min_array_value[0]; min_array_value+=jump;
	double min_y1=min_array_value[0]; min_array_value+=jump;
	
	//y dev
	double max_y2_0=max_y2[0];	max_y2+=jump;
	double max_y2_1=max_y2[0];	max_y2+=jump;
	
	double min_y2_0=min_y2[0];  min_y2+=jump;
	double min_y2_1=min_y2[0];	min_y2+=jump;
	
	//for ouputput
	double results=0.0;
	
	//temp result
	double max_x_step=max_x1-max_x0;
	double min_x_step=min_x1-min_x0;
	max_x_diff[tx]=0.0;
	min_x_diff[tx]=0.0;
	
	
	
	for(int x=0; x <len; x++)
	{
		//check max
		if(x>max_x1)
		{
			max_x0=max_x1;
			max_x1=max_array_index[0]; max_array_index+=jump;
			max_y0=max_y1;
			max_y1=max_array_value[0]; max_array_value+=jump;
			max_y2_0=max_y2_1;
			max_y2_1=max_y2[0];	max_y2+=jump;
			max_x_diff[tx]=x-max_x0;
			max_x_step=max_x1-max_x0;
			
		}
		
		//check min
		if(x>min_x1)
		{
			min_x0=min_x1;
			min_x1=min_array_index[0]; min_array_index+=jump;
			min_y0=min_y1;
			min_y1=min_array_value[0]; min_array_value+=jump;
			min_y2_0= min_y2_1 ;
			min_y2_1 = min_y2[0];	min_y2+=jump;
			min_x_diff[tx]=x-min_x0;
			min_x_step=min_x1-min_x0;
		}
		
		//compute max curve
		double B=max_x_diff[tx]/max_x_step;
		double A=1.0-B;
		results=A*max_y0+B*max_y1
			  + ((A*A*A-A)*max_y2_0 + (B*B*B-B)*max_y2_1)*max_x_step*max_x_step*one_over_6;
	//	max_y=max_y0;
			 
		
		B=min_x_diff[tx]/min_x_step;
		A=1.0-B;
		results+=A*min_y0+B*min_y1
			  + ((A*A*A-A)*min_y2_0 + (B*B*B-B)*min_y2_1)*min_x_step*min_x_step*one_over_6;
	//	min_y=min_y0;
		
		output[0] = input[0]- results*0.5;
	//	output[0] = max_y;
		input+=jump;		
		output+=jump;
		
		max_x_diff[tx]+=1.0;
		min_x_diff[tx]+=1.0;
	}
	
}



__global__ void spline_interp_uplow(double *up,
                                double *low,
								const double *input,
								const double * max_y2, const double * min_y2,
								const double * max_array_value,const double * min_array_value,
								const int *max_array_index,const int *min_array_index,
								const int len,
								const int high_stride,
								const int jump
								)
{
// 	const int tx=threadIdx.x;
// 	const int ix=blockIdx.y*high_stride+blockIdx.x*blockDim.x+tx;
	const int tx=threadIdx.x;
	const int ix = tx*high_stride+blockIdx.x*blockDim.x*high_stride;

	max_array_value+=ix;
	min_array_value+=ix;
	max_array_index+=ix;
	min_array_index+=ix;
	max_y2+=ix;
	min_y2+=ix;
	up+=ix;
    low+=ix;
	input+=ix;
	
	__shared__ double one_over_6;
	__shared__ double max_x_diff[THREAD_MAIN];
	__shared__ double min_x_diff[THREAD_MAIN];
	
	if(tx<1)
	{
		one_over_6=1.0/6.0;
	}
	__syncthreads();
	
	//x ids;
	int max_x0=max_array_index[0]; max_array_index+=jump;
	int max_x1=max_array_index[0]; max_array_index+=jump;
	
	int min_x0=min_array_index[0]; min_array_index+=jump;
	int min_x1=min_array_index[0]; min_array_index+=jump;
	
	//y value
	double max_y0=max_array_value[0]; max_array_value+=jump;
	double max_y1=max_array_value[0]; max_array_value+=jump;
	
	double min_y0=min_array_value[0]; min_array_value+=jump;
	double min_y1=min_array_value[0]; min_array_value+=jump;
	
	//y dev
	double max_y2_0=max_y2[0];	max_y2+=jump;
	double max_y2_1=max_y2[0];	max_y2+=jump;
	
	double min_y2_0=min_y2[0];  min_y2+=jump;
	double min_y2_1=min_y2[0];	min_y2+=jump;
	
	//for ouputput
	double results_1=0.0; //up
	double results_2=0.0; //out
	//temp result
	double max_x_step=max_x1-max_x0;
	double min_x_step=min_x1-min_x0;
	max_x_diff[tx]=0.0;
	min_x_diff[tx]=0.0;
	
	
	
	for(int x=0; x <len; x++)
	{
		//check max
		if(x>max_x1)
		{
			max_x0=max_x1;
			max_x1=max_array_index[0]; max_array_index+=jump;
			max_y0=max_y1;
			max_y1=max_array_value[0]; max_array_value+=jump;
			max_y2_0=max_y2_1;
			max_y2_1=max_y2[0];	max_y2+=jump;
			max_x_diff[tx]=x-max_x0;
			max_x_step=max_x1-max_x0;
			
		}
		
		//check min
		if(x>min_x1)
		{
			min_x0=min_x1;
			min_x1=min_array_index[0]; min_array_index+=jump;
			min_y0=min_y1;
			min_y1=min_array_value[0]; min_array_value+=jump;
			min_y2_0= min_y2_1 ;
			min_y2_1 = min_y2[0];	min_y2+=jump;
			min_x_diff[tx]=x-min_x0;
			min_x_step=min_x1-min_x0;
		}
		
		//compute max curve
		double B=max_x_diff[tx]/max_x_step;
		double A=1.0-B;
		results_1=A*max_y0+B*max_y1
			  + ((A*A*A-A)*max_y2_0 + (B*B*B-B)*max_y2_1)*max_x_step*max_x_step*one_over_6;
	//	max_y=max_y0;
			 
		
		B=min_x_diff[tx]/min_x_step;
		A=1.0-B;
		results_2=A*min_y0+B*min_y1
			  + ((A*A*A-A)*min_y2_0 + (B*B*B-B)*min_y2_1)*min_x_step*min_x_step*one_over_6;
	//	min_y=min_y0;
		
		up[0] = results_1;
        low[0] = results_2;
	//	output[0] = max_y;
		input+=jump;		
		up+=jump;
        low+=jump;
		
		max_x_diff[tx]+=1.0;
		min_x_diff[tx]+=1.0;
	}
	
}
//////////////////////////////////////////////////
///  if dH x L x N
///	 threadIdx.x is for N (only partial)
///  blockIdx.x is for N (segment #)
///  blockIdx.y is for dH  (stride #)
///  len is L
///  high_stride is L*N
///  jump is N
///  if H x dL x N
///	 threadIdx.x is for N (only partial)
///  blockIdx.x is for N (segment #)
///  blockIdx.y is for dL  (stride #)
///  len is H
///  high_stride is N
///  jump is dL x N
////////////////////////////////////////////////////
__global__ void spline_interp(double *output,
							  const double * max_y2, const double * min_y2,
							  const double * max_array_value,const double * min_array_value,
							  const int *max_array_index,const int *min_array_index,
							  const int len,
							  const int high_stride,
							  const int jump
							)
{
// 	const int tx=threadIdx.x;
// 	const int ix=blockIdx.y*high_stride+blockIdx.x*blockDim.x+tx;
	const int tx=threadIdx.x;
	const int ix = tx*high_stride+blockIdx.x*blockDim.x*high_stride;

	max_array_value+=ix;
	min_array_value+=ix;
	max_array_index+=ix;
	min_array_index+=ix;
	max_y2+=ix;
	min_y2+=ix;
	output+=ix;
	
	__shared__ double one_over_6;
	__shared__ double max_x_diff[THREAD_MAIN];
	__shared__ double min_x_diff[THREAD_MAIN];
	
	if(tx<1)
	{
		one_over_6=1.0/6.0;
	}
	__syncthreads();
	
	
	
	//x ids;
	int max_x0=max_array_index[0]; max_array_index+=jump;
	int max_x1=max_array_index[0]; max_array_index+=jump;
	
	int min_x0=min_array_index[0]; min_array_index+=jump;
	int min_x1=min_array_index[0]; min_array_index+=jump;
	
	//y value
	double max_y0=max_array_value[0]; max_array_value+=jump;
	double max_y1=max_array_value[0]; max_array_value+=jump;
	
	double min_y0=min_array_value[0]; min_array_value+=jump;
	double min_y1=min_array_value[0]; min_array_value+=jump;
	
	//y dev
	double max_y2_0=max_y2[0];	max_y2+=jump;
	double max_y2_1=max_y2[0];	max_y2+=jump;
	double min_y2_0=min_y2[0];   min_y2+=jump;
	double min_y2_1=min_y2[0];	min_y2+=jump;
	
	//for ouputput
	double results;

	
	//temp result
	double max_x_step=max_x1-max_x0;
	double min_x_step=min_x1-min_x0;
	 max_x_diff[tx]=0.0;
	 min_x_diff[tx]=0.0;

	
	
	for(int x=0; x <len; x++)
	{
		//check max
		if(x>max_x1)
		{
			max_x0=max_x1;
			max_x1=max_array_index[0]; max_array_index+=jump;
			max_y0=max_y1;
			max_y1=max_array_value[0]; max_array_value+=jump;
			max_y2_0=max_y2_1;
			max_y2_1=max_y2[0];	max_y2+=jump;
			max_x_diff[tx]=x-max_x0;
			max_x_step=max_x1-max_x0;
			
		}
		
		//check min
		if(x>min_x1)
		{
			min_x0=min_x1;
			min_x1=min_array_index[0]; min_array_index+=jump;
			min_y0=min_y1;
			min_y1=min_array_value[0]; min_array_value+=jump;
			min_y2_0=min_y2_1;
			min_y2_1=min_y2[0];	min_y2+=jump;
			min_x_diff[tx]=x-min_x0;
			min_x_step=min_x1-min_x0;
		}
		
		//compute max curve
		double B=max_x_diff[tx]/max_x_step;
		double A=1.0-B;
		results=A*max_y0+B*max_y1
			  + ((A*A*A-A)*max_y2_0 + (B*B*B-B)*max_y2_1)*max_x_step*max_x_step*one_over_6;
		
		B=min_x_diff[tx]/min_x_step;
		A=1.0-B;
		results+=A*min_y0+B*min_y1
			  + ((A*A*A-A)*min_y2_0 + (B*B*B-B)*min_y2_1)*min_x_step*min_x_step*one_over_6;
		
		output[0] -= results*0.5;	output+=jump;
		
		max_x_diff[tx]+=1.0;
		min_x_diff[tx]+=1.0;
	}
	
}

/////////////////////////////////////////
// Use d_current (the IMF after several siftings) to
//  (0) Average all d_current --> sh_reduction
//  (1) Store sh_reduction as IMF by multiply the std with d_current
//  (2) Update d_ensemble by reducting d_current from d_ensemble
//
///  dH x L x N   ->  dH x L x N   and H x L_padding
///  threadIdx.x for N  on input/output
///              for L  on imf
///  threadIdx.y for L  on input/output
///  blockIdx.x for L segment / threadIdx.x
///  blockIdx.y for dH segment
///  high_stride is L
///  NOTE: threadIdx.x == threadIdx.y
////////////////////////////
__global__ void update_imf_lowbit(
						  double *imf,        //shifted index for this imf -> an output
						  double *output,     // d_ensemble -> 
						  const double *input, // d_current
						  const int n_len,     // 2*ne
						  const int high_stride,  //L
						  const int L_padding,
						  const int L_len,
						  const int bias
						  )
{
	const int tx=threadIdx.x;   
	const int ty=threadIdx.y;
	
	__shared__ double sh_reduction[UPDATE_IMF_SHARED_SIZE];  // square 16*16 or 32*32 

	const int s_id = tx*blockDim.y+ty;   // for shared memory, transported
	const int id = ty*blockDim.x+tx;   // for shared memory
	sh_reduction[id]=0.0;
	
	
	
	const int bx = blockIdx.x * blockDim.x;
	const int by = blockIdx.y * high_stride;
	const int r_times = n_len/blockDim.x;
	double std = CONST_STD[blockIdx.y+bias];
	std = std >EPS ? std/(double)n_len : 1.0/(double)n_len;  // This is actually the 2*ne

	input += (bx+ ty)*n_len +by+ tx;
	output += (bx+ ty)*n_len +by+ tx;
	imf += blockIdx.y *L_padding +bx+tx;
	__syncthreads();
	
	if(bx+ty<L_len)
	{
		for(int i=0;i<r_times;i++)
		{
			//load and copy
			double v=input[0]; input+=blockDim.x;
			sh_reduction[s_id] += v;
			output[0]-= v; output+=blockDim.x;
			
		}
	}
	
	__syncthreads();
	
	//reduction
#if UPDATE_IMF_SHARED_SIZE>512
	if(id<512) {sh_reduction[id]+=sh_reduction[id+512];}	__syncthreads();
#endif
#if UPDATE_IMF_SHARED_SIZE>256
	if(id<256) {sh_reduction[id]+=sh_reduction[id+256];}	__syncthreads();
#endif
	if(id<128) {sh_reduction[id]+=sh_reduction[id+128];}	__syncthreads();
	if(id<64) {sh_reduction[id]+=sh_reduction[id+64];}	__syncthreads();
	if(id<32) {sh_reduction[id]+=sh_reduction[id+32];}	//__syncthreads();
	if(id<16) {
					sh_reduction[id]+=sh_reduction[id+16];
					imf[0]+=sh_reduction[id]*std;
			   }
	
}
/////////////////////////////////////////
///  dH x L x N   ->  dH x L x N   and H x L_padding
///  threadIdx.x for N  on input/output
///              for L  on imf
///  threadIdx.y for L  on input/output
///  blockIdx.x for L segment / threadIdx.x
///  blockIdx.y for dH segment
///  high_stride is L
///  NOTE: threadIdx.x == threadIdx.y
////////////////////////////
__global__ void update_imf_trend_lowbit(
						  double *imf_ensemble,
						  double *imf_current,
						  const double *ensemble, 
						  const double *current,
						  const int n_len,
						  const int high_stride,  //L
						  const int L_padding,
						  const int L_len,
						  const int bias
						  )
{
	const int tx=threadIdx.x;   
	const int ty=threadIdx.y;
	
	__shared__ double sh_reduction[2][UPDATE_IMF_SHARED_SIZE];  // square 16*16 or 32*32 

	const int s_id = tx*blockDim.y+ty;   // for shared memory, transported
	const int id = ty*blockDim.x+tx;   // for shared memory
	sh_reduction[0][id]=0.0;
	sh_reduction[1][id]=0.0;
	
	
	const int bx = blockIdx.x * blockDim.x;
	const int by = blockIdx.y * high_stride;
	const int r_times = n_len/blockDim.x;
	double std = CONST_STD[blockIdx.y+bias];
	std = std >EPS ? std/(double)n_len : 1.0/(double)n_len;   
	ensemble += (bx+ ty)*n_len +by+ tx;
	current += (bx+ ty)*n_len + by+tx;
	imf_ensemble += bx + blockIdx.y *L_padding +tx;
	imf_current += blockIdx.y *L_padding +bx+tx;
	__syncthreads();
	
	if(bx+ty<L_len)
	{
		for(int i=0;i<r_times;i++)
		{
			//only load
			double v=current[0]; current+=blockDim.x;
			sh_reduction[0][s_id] += v;
			sh_reduction[1][s_id] += ensemble[0]-v;   ensemble+=blockDim.x;
			
		}
	}
	
	__syncthreads();
	
	//reduction
#if UPDATE_IMF_SHARED_SIZE>512
	if(id<512) {
		sh_reduction[0][id]+=sh_reduction[0][id+512];
		sh_reduction[1][id]+=sh_reduction[1][id+512];
	}	
	__syncthreads();
#endif
#if UPDATE_IMF_SHARED_SIZE>256
	if(id<256) {
		sh_reduction[0][id]+=sh_reduction[0][id+256];
		sh_reduction[1][id]+=sh_reduction[1][id+256];
	}	
	__syncthreads();
#endif
	if(id<128) {
		sh_reduction[0][id]+=sh_reduction[0][id+128];
		sh_reduction[1][id]+=sh_reduction[1][id+128];
	}	
	__syncthreads();
	if(id<64) {
		sh_reduction[0][id]+=sh_reduction[0][id+64];
		sh_reduction[1][id]+=sh_reduction[1][id+64];
	}	
	__syncthreads();
	if(id<32) {
		sh_reduction[0][id]+=sh_reduction[0][id+32];
		sh_reduction[1][id]+=sh_reduction[1][id+32];
	}	
	__syncthreads();
	if(id<16) {
				sh_reduction[0][id]+=sh_reduction[0][id+16];
				sh_reduction[1][id]+=sh_reduction[1][id+16];
				imf_current[0]+=sh_reduction[0][id]*std;
				imf_ensemble[0]+=sh_reduction[1][id]*std;
			   }
	
}
/////////////////////////////////////////
///  H x dL x N   ->  H x dL x N   and H x L_padding
///  threadIdx.x for N  on input/output
///              for dL  on imf
///  threadIdx.y for dL  on input/output
///  blockIdx.x for L segment / threadIdx.x
///  blockIdx.y for H segment
///  high_stride is dL
///  NOTE: threadIdx.x == threadIdx.y
////////////////////////////
__global__ void update_imf(
						  double *imf,        //imf
						  double *output,     // ensemble
						  const double *input, //current
						  const int n_len,
						  const int high_stride,
						  const int L_padding,
						  const int L_len,
						  const int bias
						  )
{
	const int tx=threadIdx.x;   // x
	const int ty=threadIdx.y;   // y
	
	__shared__ double sh_reduction[UPDATE_IMF_SHARED_SIZE];  // square 16*16 or 32*32 

	const int s_id = tx*blockDim.y+ty;   // for shared memory, transported
	const int id = ty*blockDim.x+tx;   // for shared memory, 
	sh_reduction[id]=0.0;
	
	
	const int bx = blockIdx.x * blockDim.x;  //
	const int by = blockIdx.y * high_stride;
	const int r_times = n_len/blockDim.x;
	double std = CONST_STD[bx+tx+bias];
	//double std = CONST_STD[blockIdx.x];
	std = std >EPS ? std/(double)n_len : 1.0/(double)n_len;  
	input += (bx+by + ty)*n_len + tx;
	output += (bx+by + ty)*n_len + tx;
	imf += blockIdx.y *L_padding +bx+tx;
	__syncthreads();
	
	if(bx+ty<L_len)
	{
		for(int i=0;i<r_times;i++)
		{
			//load and copy
			double v=input[0]; input+=blockDim.x;
			sh_reduction[s_id] += v;
			output[0]-= v; output+=blockDim.x;
			
		}
	}
	
	__syncthreads();
	
	//reduction
#if UPDATE_IMF_SHARED_SIZE>512
	if(id<512) {sh_reduction[id]+=sh_reduction[id+512];}	__syncthreads();
#endif
#if UPDATE_IMF_SHARED_SIZE>256
	if(id<256) {sh_reduction[id]+=sh_reduction[id+256];}	__syncthreads();
#endif
	if(id<128) {sh_reduction[id]+=sh_reduction[id+128];}	__syncthreads();
	if(id<64) {sh_reduction[id]+=sh_reduction[id+64];}	__syncthreads();
	if(id<32) {sh_reduction[id]+=sh_reduction[id+32];}	__syncthreads();
	if(id<16) {
				sh_reduction[id]+=sh_reduction[id+16];
				imf[0]+=sh_reduction[id]*std;
			//	imf[0]+=sh_reduction[id];
			   }
	
}

/////////////////////////////////////////
///  H x dL x N   ->  H x dL x N   and H x L_padding
///  threadIdx.x for N  on input/output
///              for dL  on imf
///  threadIdx.y for dL  on input/output
///  blockIdx.x for L segment / threadIdx.x
///  blockIdx.y for H segment
///  high_stride is dL
///  NOTE: threadIdx.x == threadIdx.y
////////////////////////////
//for last one  (trend and IMF last)
__global__ void update_imf_trend(
						  double *imf_ensemble,
						  double *imf_current,
						  const double *ensemble, 
						  const double *current,
						  const int n_len,
						  const int high_stride,
						  const int L_padding,
						  const int L_len,
						  const int bias
						  )
{
	const int tx=threadIdx.x;   
	const int ty=threadIdx.y;
	
	__shared__ double sh_reduction[2][UPDATE_IMF_SHARED_SIZE];  // square 16*16 or 32*32 

	const int s_id = tx*blockDim.y+ty;   // for shared memory, transported
	const int id = ty*blockDim.x+tx;   // for shared memory, transported
	sh_reduction[0][id]=0.0;
	sh_reduction[1][id]=0.0;
	
	
	const int bx = blockIdx.x * blockDim.x;
	const int by = blockIdx.y * high_stride;
	const int r_times = n_len/blockDim.x;
	double std = CONST_STD[bx+tx+bias];
//	double std = CONST_STD[blockIdx.x];
	std = std >EPS ? std/(double)n_len : 1.0/(double)n_len;  
	
	current += (bx+by + ty)*n_len + tx;
	ensemble += (bx+by + ty)*n_len + tx;
	imf_current += bx + blockIdx.y *L_padding +tx;
	imf_ensemble += bx + blockIdx.y *L_padding +tx;
	__syncthreads();
	
	if(bx+ty<L_len)
	{
		for(int i=0;i<r_times;i++)
		{
			//only load
			double v=current[0]; current+=blockDim.x;
			sh_reduction[0][s_id] += v;
			sh_reduction[1][s_id] += ensemble[0]-v;   ensemble+=blockDim.x;
		}
	}
	
	__syncthreads();
	
	//reduction
#if UPDATE_IMF_SHARED_SIZE>512
	if(id<512) {
		sh_reduction[0][id]+=sh_reduction[0][id+512];
		sh_reduction[1][id]+=sh_reduction[1][id+512];
	}	
	__syncthreads();
#endif
#if UPDATE_IMF_SHARED_SIZE>256
	if(id<256) {
		sh_reduction[0][id]+=sh_reduction[0][id+256];
		sh_reduction[1][id]+=sh_reduction[1][id+256];
	}	
	__syncthreads();
#endif
	if(id<128) {
		sh_reduction[0][id]+=sh_reduction[0][id+128];
		sh_reduction[1][id]+=sh_reduction[1][id+128];
	}	
	__syncthreads();
	if(id<64) {
		sh_reduction[0][id]+=sh_reduction[0][id+64];
		sh_reduction[1][id]+=sh_reduction[1][id+64];
	}	
	__syncthreads();
	if(id<32) {
		sh_reduction[0][id]+=sh_reduction[0][id+32];
		sh_reduction[1][id]+=sh_reduction[1][id+32];
	}	
	
	if(id<16) {
		sh_reduction[0][id]+=sh_reduction[0][id+16];
		sh_reduction[1][id]+=sh_reduction[1][id+16];
		imf_current[0]+=sh_reduction[0][id]*std;
		imf_ensemble[0]+=sh_reduction[1][id]*std;

	 }
	
}

/////////////////////////////////////////
// Use d_current (the IMF after several siftings) to
//  (0) Average all d_current --> sh_reduction
//  (1) Store sh_reduction as IMF by multiply the std with d_current
//  (2) Update d_ensemble by reducting d_current from d_ensemble
//
///  dH x L x N   ->  dH x L x N   and H x L_padding
///  threadIdx.x for N  on input/output
///              for L  on imf
///  threadIdx.y for L  on input/output
///  blockIdx.x for L segment / threadIdx.x
///  blockIdx.y for dH segment
///  high_stride is L
///  NOTE: threadIdx.x == threadIdx.y
////////////////////////////
__global__ void update_x0(
    double *output,     // d_ensemble -> 
    const double *input, // d_current
    const int high_stride,  //L
    const int L_len,
    const int jump
    )
{
const int tx=threadIdx.x;   
//const int ty=threadIdx.y;

///__shared__ double sh_reduction[UPDATE_IMF_SHARED_SIZE];  // square 16*16 or 32*32 

///const int s_id = tx*blockDim.y+ty;   // for shared memory, transported
///const int id = ty*blockDim.x+tx;   // for shared memory
///sh_reduction[id]=0.0;


///__syncthreads();
///const int bx = blockIdx.x * blockDim.x;
const int bx = blockIdx.x * blockDim.x;
///const int by = blockIdx.y * high_stride;
///const int by = blockIdx.y * blockDim.y;

//const int r_times = L_padding/blockDim.y;
//int by;
//double std = CONST_STD[blockIdx.y+bias];
//std = std >EPS ? std/(double)n_len : 1.0/(double)n_len;  // This is actually the 2*ne

input += (bx + tx)*high_stride;
output += (bx + tx)*high_stride;

//imf += (bx + tx)*high_stride + by + ty;


///(2) Update d_ensemble by reducting d_current from d_ensemble
for(int i=0;i<L_len;i++)
{
//double v=input[0]; 
double v=input[0];  input += jump;//for debug
output[0]-= v;		  output += jump;
}
/*
for(int i=0;i<r_times;i++)
{
by = i*blockDim.y;
if(bx+ty<L_len)
{
//load and copy
input += (bx + tx)*high_stride + by + ty;
output += (bx + tx)*high_stride + by + ty;
//double v=input[0]; 
double v=input[0]+1;  //for debug
output[0]-= v;

}
}
*/

} 