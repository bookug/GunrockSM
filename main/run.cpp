/*=============================================================================
# Filename: dig.cpp
# Author: Bookug Lobert 
# Mail: 1181955272@qq.com
# Last Modified: 2016-10-24 14:39
# Description: 
=============================================================================*/

#include "../util/Util.h"
#include "../io/IO.h"
#include "../graph/Graph.h"
#include "../match/Match.cuh"
#include <cuda_runtime.h>

using namespace std;

//NOTICE:a pattern occurs in a graph, then support++(not the matching num in a graph), support/N >= minsup
vector<Graph*> query_list;

int
main(int argc, const char * argv[])
{
	int i, j, k;

	string output = "ans.txt";
	if(argc > 4 || argc < 3)
	{
		cerr<<"invalid arguments!"<<endl;
		return -1;
	}
	string data = argv[1];
	string query = argv[2];
	if(argc == 4)
	{
		output = argv[3];
	}

	cerr<<"args all got!"<<endl;
	
	 int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0) {
            fprintf(stderr, "error: no devices supporting CUDA.\n");
            exit(EXIT_FAILURE);
        }
        int dev = 1;
        cudaSetDevice(dev);

        cudaDeviceProp devProps;
        if (cudaGetDeviceProperties(&devProps, dev) == 0)
        {
            printf("Using device %d:\n", dev);
            printf("%s; global mem: %luB; compute v%d.%d; clock: %d kHz; shared mem: %dB; block threads: %d; SM count: %d\n",
                   devProps.name, devProps.totalGlobalMem,
                   (int)devProps.major, (int)devProps.minor,
                   (int)devProps.clockRate,
                           devProps.sharedMemPerBlock, devProps.maxThreadsPerBlock, devProps.multiProcessorCount);
        }
//	setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1);
        cout<<"GPU selected"<<endl;

	
	long t1 = Util::get_cur_time();

	IO io = IO(query, data, output);
	//read query file and keep all queries in memory
	io.input(query_list);
	int qnum = query_list.size();
	
	cerr<<"input ok!"<<endl;
	long t2 = Util::get_cur_time();

	Graph* data_graph = NULL;
	while(true)
	{
		if(!io.input(data_graph))
		{
			break;
		}
		for(i = 0; i < qnum; ++i)
		{
//			float elapsed_time = 0.0f;
			long af = Util::get_cur_time();
			Match m(query_list[i], data_graph);
			printf("the match class build done\n");
			io.output(i);
		//	cudaEvent_t start, stop;
		//	cudaEventCreate(&start);
		//	cudaEventCreate(&stop);
		//	cudaEventRecord(start,0);
			m.match(io);
		//	cudaEventRecord(stop,0);
		//	cudaEventSynchronize(start);
		//	cudaEventSynchronize(stop);
		//	cudaEventElapsedTime(&elapsed_time, start, stop);
		//
	//		printf("the match process use %f ms\n\n\n",af-bf);
			io.output();
			io.flush();
			long bf = Util::get_cur_time();
			cerr<<"the match process use "<<bf - af<<" ms."<<endl;
		}
		delete data_graph;
	}

	cerr<<"match ended!"<<endl;
	long t3 = Util::get_cur_time();

	//output the time for contrast
	cerr<<"part 1 used: "<<(t2-t1)<<"ms"<<endl;
	cerr<<"part 2 used: "<<(t3-t2)<<"ms"<<endl;
	cerr<<"total time used: "<<(t3-t1)<<"ms"<<endl;

	//release all and flush cached writes
	for(i = 0; i < qnum; ++i)
	{
		delete query_list[i];
	}
	io.flush();

	return 0;
}

