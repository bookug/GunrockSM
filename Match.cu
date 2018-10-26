/*=============================================================================
# Filename: Match.cpp
# Author: Bookug Lobert 
# Mail: 1181955272@qq.com
# Last Modified: 2016-12-15 01:38
# Description: 
=============================================================================*/

#include "Match.cuh"
#include <cuda_runtime.h>

#define MAXTHREADPERBLOCK 512
#define MAXTOTALTHREAD 1000000
using namespace std;


Match::Match(Graph* _query, Graph* _data)
{
	printf("enter match here\n");
	this->query = _query;
	this->data = _data;
	this->qsize = _query->vSize();
	this->dsize = _data->vSize();

	//printf("query graph processed\n\n\n");
	//move csr to GPU
	this->qTotalInNum = _query->totalInNum;
	this->qTotalOutNum = _query->totalOutNum;

	this->dTotalInNum = _data->totalInNum;
	this->dTotalOutNum = _data->totalOutNum;
	
	cudaMalloc(&qlabels, sizeof(int)*qsize);
	cudaMalloc(&qInRowOffset, sizeof(int)*(qsize+1));
	cudaMalloc(&qOutRowOffset, sizeof(int)*(1+qsize));
	cudaMalloc(&qInValues,sizeof(int)*qTotalInNum);
	cudaMalloc(&qInColOffset,sizeof(int)*qTotalInNum);
	cudaMalloc(&qOutValues,sizeof(int)*qTotalOutNum);
	cudaMalloc(&qOutColOffset,sizeof(int)*qTotalOutNum);
	
	cudaMalloc(&dlabels, sizeof(int)*dsize);
	cudaMalloc(&dInRowOffset, sizeof(int)*(dsize+1));
	cudaMalloc(&dOutRowOffset, sizeof(int)*(1+dsize));
	cudaMalloc(&dInValues,sizeof(int)*dTotalInNum);
	cudaMalloc(&dInColOffset,sizeof(int)*dTotalInNum);
	cudaMalloc(&dOutValues,sizeof(int)*dTotalOutNum);
	cudaMalloc(&dOutColOffset,sizeof(int)*dTotalOutNum);
	
//	printf("malloc in build match all done\n");
//	printf("the qsize is %d,and the query labels is \n",qsize);
//	for(int i = 0; i < qsize; i++)
//		printf("the qlabel %d is %d\n",i,_query->labels[i]);	
	cudaMemcpy((void *)qlabels, (void *)_query->labels, sizeof(int)*qsize, cudaMemcpyHostToDevice);
	cudaMemcpy((void *)qInRowOffset, (void *)_query->inRowOffset, sizeof(int)*(qsize+1), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)qOutRowOffset, (void *)_query->outRowOffset, sizeof(int)*(qsize+1), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)qInColOffset, (void *)_query->inColOffset, sizeof(int)*qTotalInNum, cudaMemcpyHostToDevice);
	cudaMemcpy((void *)qOutColOffset, (void *)_query->outColOffset, sizeof(int)*qTotalOutNum, cudaMemcpyHostToDevice);
	cudaMemcpy((void *)qInValues, (void *)_query->inValues, sizeof(int)*qTotalInNum, cudaMemcpyHostToDevice);
	cudaMemcpy((void *)qOutValues, (void *)_query->outValues, sizeof(int)*qTotalOutNum, cudaMemcpyHostToDevice);

	cudaMemcpy((void *)dlabels, (void *)_data->labels, sizeof(int)*dsize, cudaMemcpyHostToDevice);
	cudaMemcpy((void *)dInRowOffset, (void *)_data->inRowOffset, sizeof(int)*(dsize+1), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)dOutRowOffset, (void *)_data->outRowOffset, sizeof(int)*(dsize+1), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)dInColOffset, (void *)_data->inColOffset, sizeof(int)*dTotalInNum, cudaMemcpyHostToDevice);
	cudaMemcpy((void *)dOutColOffset, (void *)_data->outColOffset, sizeof(int)*dTotalOutNum, cudaMemcpyHostToDevice);
	cudaMemcpy((void *)dInValues, (void *)_data->inValues, sizeof(int)*dTotalInNum, cudaMemcpyHostToDevice);
	cudaMemcpy((void *)dOutValues, (void *)_data->outValues, sizeof(int)*dTotalOutNum, cudaMemcpyHostToDevice);
	//TODO:
	//cudaDeviceSynchronize();
	printf("move csr to GPU done\n");
	//cuda init
	int t_0,t_1;
    t_0 = Util::get_cur_time();
	int* warmup = NULL;
	cudaMalloc(&warmup, sizeof(int));
	cudaFree(warmup);
	cout<<"GPU warmup finished"<<endl;
	size_t size = 0x7fffffff;
	/*size *= 3;   //heap corruption for 3 and 4*/
	size *= 5;
	/*size *= 2;*/
	//NOTICE: the memory alloced by cudaMalloc is different from the GPU heap(for new/malloc in kernel functions)
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);
	cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
	cout<<"check heap limit: "<<size<<endl;	
	t_1 = Util::get_cur_time();
    printf("cuda init done,it uses %d ms\n",t_1 - t_0);

}

Match::~Match()
{

	for(int i = 0; i < qsize; i ++)
		delete [] nodeCans[i];
	for(int i = 0; i < query->edgeSize; i ++)
		delete [] edgeCans[i];
	delete [] nodeCansSize;
	delete [] edgeCansSize;
	delete [] nodeCans;
	delete [] edgeCans;
}

//Two States of core queues: matching [0,size) not-matching -1
//Four States of in/out queues: matching depth   in_queue depth  out_queue depth other -1
//NOTICE:a vertex maybe appear in in_queue and out_queue at the same time
__global__ void findEgdeCansInDevice1(int label,
									  int* toDevice,
									  int nodeCansSize1,
									  int nodeCansSize2,
									  int totalThreadNum,
									  int* d_flagList,
									  d_Match* d_m)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= totalThreadNum)
		return;
	bool reverse = false;
	int canSize = nodeCansSize2;
	if(nodeCansSize1 < nodeCansSize2){
		reverse = true;
		canSize = nodeCansSize1;
	}
	int* cans1;
	int* cans2;
	if(reverse)
	{
		cans1 = toDevice + nodeCansSize1;
		cans2 = toDevice;
	}else{
		cans1 = toDevice;
		cans2 = toDevice + nodeCansSize1;
	}
	int ansNum = 0;
	int curId = cans1[idx];
	int* adjList;
	int* adjLable;
	int adjListSize;
	if(reverse)
	{
		adjList = d_m->dOutColOffset + d_m->dOutRowOffset[curId];
		adjLable = d_m->dOutValues + d_m->dOutRowOffset[curId];
		adjListSize = d_m->dOutRowOffset[curId+1] - d_m->dOutRowOffset[curId];
	}else{
		adjList = d_m->dInColOffset + d_m->dInRowOffset[curId];
		adjLable = d_m->dInValues + d_m->dInRowOffset[curId];
		adjListSize = d_m->dInRowOffset[curId+1] - d_m->dInRowOffset[curId];
	}
	int i = 0,j =0;
	while(i < canSize && j < adjListSize)
	{
		if(cans2[i] == adjList[j])
		{
			if(adjLable[j] == label)
				ansNum ++;
			i++;
			j++;
		}
		else if(cans2[i] < adjList[j])
		{
			i++;
		}
		else{
			j++;
		}
	}
	d_flagList[idx] = ansNum;
	return ;

}

__global__ void findEgdeCansInDevice2(int label,
									  int* toDevice,
									  int nodeCansSize1,
									  int nodeCansSize2,
									  int totalThreadNum,
									  int* d_flagList,
									  int* ansArea,
									  d_Match* d_m)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= totalThreadNum)
		return;
	bool reverse = false;
	int canSize = nodeCansSize2;
	int cansWritePos = 1;
	if(nodeCansSize1 < nodeCansSize2){
		reverse = true;
		canSize = nodeCansSize1;
		cansWritePos = 0;
	}
	int* cans1;
	int* cans2;
	if(reverse)
	{
		cans1 = toDevice + nodeCansSize1;
		cans2 = toDevice;
	}else{
		cans1 = toDevice;
		cans2 = toDevice + nodeCansSize1;
	}
	int startPos = d_flagList[idx];
	int curId = cans1[idx];
	int* adjList;
	int* adjLable;
	int adjListSize;
	if(reverse)
	{
		adjList = d_m->dOutColOffset + d_m->dOutRowOffset[curId];
		adjLable = d_m->dOutValues + d_m->dOutRowOffset[curId];
		adjListSize = d_m->dOutRowOffset[curId+1] - d_m->dOutRowOffset[curId];
	}else{
		adjList = d_m->dInColOffset + d_m->dInRowOffset[curId];
		adjLable = d_m->dInValues + d_m->dInRowOffset[curId];
		adjListSize = d_m->dInRowOffset[curId+1] - d_m->dInRowOffset[curId];
	}
	int i = 0,j =0;
	while(i < canSize && j < adjListSize)
	{
		if(cans2[i] == adjList[j])
		{
			if(adjLable[j] == label){
				ansArea[startPos*2+cansWritePos] = cans2[i];
				ansArea[startPos*2+1- cansWritePos] = curId;
				startPos ++;
			}
			i++;
			j++;
		}
		else if(cans2[i] < adjList[j])
		{
			i++;
		}
		else{
			j++;
		}
	}
	return ;	
}

__global__ void edgeMerge1(int* d_edgeCansOff,
						   int* d_edgeCans,
						   int* d_flagList,
						   int totalThreadNum,
						   int qsize,
						   d_Match* d_m,
						   int startPos,
						   int edgeNum,
						   int * d_matchResult)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= totalThreadNum)
		return;
//	if(idx%MAXTHREADPERBLOCK== 0)
//		printf("here is idx 0 from %d\n",blockIdx.x);
	int* aMatch = d_matchResult + idx*qsize;
	idx = idx + startPos;
	int mul = 1;
	int edgeCount = edgeNum - 1;
	bool prune = false;
	for(int i = qsize-1; i >= 0; i--)
	{

		for(int j = d_m->qInRowOffset[i+1]-1; j >= d_m->qInRowOffset[i]; j --)
		{
			int n1 = d_edgeCans[(d_edgeCansOff[edgeCount] + (idx/mul)%(d_edgeCansOff[edgeCount+1] - d_edgeCansOff[edgeCount]))*2];
			int n2 = d_edgeCans[(d_edgeCansOff[edgeCount] + (idx/mul)%(d_edgeCansOff[edgeCount+1] - d_edgeCansOff[edgeCount]))*2+1];
			//if(edgeCount == 0)
			//	printf("the edgeCount is 0, the cans is %d, %d\n",n1,n2);
			mul *= (d_edgeCansOff[edgeCount+1] - d_edgeCansOff[edgeCount]);
			edgeCount --;
			int q2 = d_m->qInColOffset[j];
			if(aMatch[i] != -1)
			{
				if(aMatch[i] != n1)
				{
					prune = true;
				}
			}else{
				aMatch[i] = n1;
			}
			if(aMatch[q2] != -1)
			{
				if(aMatch[q2] != n2){
					prune = true;
				}
			}else{
				aMatch[q2] = n2;
			}
			
		}
	}
	//printf("here i get an answer!\n");
	for(int i = 0; i < qsize; i ++)
	    for(int j = i+1; j < qsize; j ++)
	    {
			if(aMatch[i] == aMatch[j])
			{
			    prune = true;
			}
	    }
	if(prune)
	{
/*		if(
		aMatch[0] == 4280
		&& aMatch[1] == 4279
		aMatch[2] == 4249
		&& aMatch[3] == 4248
		&& aMatch[4] == 1075
		&& aMatch[5] == 1072
		&& aMatch[6] == 1071
		)
			printf("%d %d %d %d %d %d %d\n",aMatch[0],aMatch[1],aMatch[2],aMatch[3],aMatch[4],aMatch[5],aMatch[6]); */
		return ;
	}
	d_flagList[idx-startPos] = 1;
	//printf("here i get an answer!\n");
	return ;

}
__global__ void edgeMerge2(int* d_edgeCansOff,
						   int* d_edgeCans,
						   int* d_flagList,
						   int effectiveSize,
						   int* d_ansArea,
						   int qsize,
						   d_Match* d_m,
						   int edgeNum)
{
	int real_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(real_idx >= effectiveSize)
		return;

	int idx = d_flagList[real_idx];
	int *writePos = d_ansArea + real_idx*qsize;
	int mul = 1;
	int edgeCount = edgeNum - 1;
	for(int i = qsize-1; i >= 0; i--)
	{
		for(int j = d_m->qInRowOffset[i+1]-1; j >= d_m->qInRowOffset[i]; j --)
		{
			int n1 = d_edgeCans[(d_edgeCansOff[edgeCount] + (idx/mul)%(d_edgeCansOff[edgeCount+1] - d_edgeCansOff[edgeCount]))*2];
			int n2 = d_edgeCans[(d_edgeCansOff[edgeCount] + (idx/mul)%(d_edgeCansOff[edgeCount+1] - d_edgeCansOff[edgeCount]))*2+1];

			mul *= (d_edgeCansOff[edgeCount+1] - d_edgeCansOff[edgeCount]);
			edgeCount --;
			
			writePos[i] = n1;
			
			writePos[d_m->qInColOffset[j]] = n2;		
		}
	}
	return ;
}
void 
Match::match(IO& io)
{
	if(qsize > dsize)
	{
		return;
	}
	d_Match * d_m;
	cudaMalloc(&d_m,sizeof(d_Match));
	cudaMemcpy((void *)d_m, (void *)this,sizeof(Match),cudaMemcpyHostToDevice);
	//find cans for query nodes
	printf("start find node cans\n");
	nodeCans = new int*[qsize];
	nodeCansSize  = new int[qsize];
/*	for(int i = 0; i <= data->nodeLabelNum; i ++)
		printf("%d\n",data->reverseLabel[i]);
	printf(" \n\n");
	for(int i = 0; i < data->reverseLabel[data->nodeLabelNum];i ++)
		printf("%d\n",data->reverseValues[i]);


	for(int i = 0; i <= query->nodeLabelNum; i ++)
                printf("%d\n",query->reverseLabel[i]);
        printf(" \n\n");
        for(int i = 0; i < query->reverseLabel[query->nodeLabelNum];i ++)
                printf("%d\n",query->reverseValues[i]);*/

	for(int i = 0; i < qsize; i ++)
	{
		int targerLabel = query->vertices[i].label - 1;
		int targetDegree = query->vertices[i].in.size() + query->vertices[i].out.size();
		int* tempCans = new int [data->reverseLabel[targerLabel+1] - data->reverseLabel[targerLabel]];
	//	printf("the can size is %d, targetLabel is %d,degree is %d\n",
//		data->reverseLabel[targerLabel+1] - data->reverseLabel[targerLabel],targerLabel,targetDegree);
		int cansPos = 0;
		for(int j = data->reverseLabel[targerLabel]; j < data->reverseLabel[targerLabel+1]; j ++)
		{
			int curId = data->reverseValues[j];
	//		printf("j is %d,curId is %d, degree is %d\n",j,curId,data->vertices[curId].in.size() + data->vertices[curId].out.size());
			if(data->vertices[curId].in.size() >= query->vertices[i].in.size()&&
				data->vertices[curId].out.size() >= query->vertices[i].out.size()){
				tempCans[cansPos++] = curId;
			}
		}
		nodeCans[i] = new int[cansPos];
		nodeCansSize[i] = cansPos;
		memcpy(nodeCans[i],tempCans,sizeof(int)*cansPos);
		delete [] tempCans;
	}
	/*for(int i = 0; i < qsize; i++)
	{
		printf("here is cans for node %d\n",i);
		for(int j = 0; j < nodeCansSize[i];j ++)
			printf("%d\n",nodeCans[i][j]);
	}*/	
	//find cans for edges
	printf("start find cans edges\n");
	edgeCans = new int*[query->edgeSize];
	edgeCansSize = new int[query->edgeSize];
	//TODO:here I use inEdge to do all things
	int edgeCount = 0;
	for(int i = 0; i < qsize; i++)
	{
		for(int j = query->inRowOffset[i]; j < query->inRowOffset[i+1]; j ++)
		{
			int targerLabel = query->inValues[j];
			int anotherNode = query->inColOffset[j];
			int* toDevice;
			int totalThreadNum = nodeCansSize[i];
			if(totalThreadNum < nodeCansSize[anotherNode])
				totalThreadNum = nodeCansSize[anotherNode];
			if(totalThreadNum == 0)
				return ;
			int* d_flagList;
			int* h_flagList = new int[totalThreadNum+1];
			memset(h_flagList,0,sizeof(int)*(1+totalThreadNum));
			cudaMalloc(&d_flagList, sizeof(int)*totalThreadNum);
			cudaMemcpy((void *)d_flagList, (void *)(h_flagList+1), sizeof(int)*totalThreadNum, cudaMemcpyHostToDevice);
			cudaMalloc(&toDevice, sizeof(int)*(nodeCansSize[i]+nodeCansSize[anotherNode]));
			cudaMemcpy((void *)toDevice,nodeCans[i],sizeof(int)*nodeCansSize[i],cudaMemcpyHostToDevice);
			cudaMemcpy((void *)(toDevice + nodeCansSize[i]),nodeCans[anotherNode],sizeof(int)*nodeCansSize[anotherNode], cudaMemcpyHostToDevice);
			findEgdeCansInDevice1<<<(totalThreadNum+MAXTHREADPERBLOCK-1)/MAXTHREADPERBLOCK,MAXTHREADPERBLOCK>>>(targerLabel,
																												toDevice,
																												nodeCansSize[i],
																												nodeCansSize[anotherNode],
																												totalThreadNum,
																												d_flagList,
																												d_m);
			cudaMemcpy((void *)(h_flagList+1),(void *)d_flagList,sizeof(int)*totalThreadNum,cudaMemcpyDeviceToHost);
			cudaFree(d_flagList);

	//		printf("here is the h_flagList\n");
	//		for(int k = 0; k <= totalThreadNum; k++)
	//			printf("h_flagList[%d] is %d\n",k,h_flagList[k]);

			for(int k = 1; k <= totalThreadNum; k ++)
			{
				h_flagList[k] = h_flagList[k] + h_flagList[k-1]; 
			}
			
			
			int* d_ansArea;
			cudaMalloc(&d_ansArea, sizeof(int)*2*h_flagList[totalThreadNum]);
			cudaMalloc(&d_flagList,sizeof(int)*(totalThreadNum+1));
			cudaMemcpy((void *)d_flagList, (void *)h_flagList, sizeof(int)*(totalThreadNum+1), cudaMemcpyHostToDevice);
			findEgdeCansInDevice2<<<(totalThreadNum+MAXTHREADPERBLOCK-1)/MAXTHREADPERBLOCK,MAXTHREADPERBLOCK>>>(targerLabel,
																												toDevice,
																												nodeCansSize[i],
																												nodeCansSize[anotherNode],
																												totalThreadNum,
																												d_flagList,
																												d_ansArea,
																												d_m);
			edgeCans[edgeCount] = new int[h_flagList[totalThreadNum]*2];
			cudaMemcpy((void *)edgeCans[edgeCount],(void *)d_ansArea,sizeof(int)*2*h_flagList[totalThreadNum],cudaMemcpyDeviceToHost);
			edgeCansSize[edgeCount++] = h_flagList[totalThreadNum];

			cudaFree(d_flagList);
			cudaFree(toDevice);
			cudaFree(d_ansArea);
			delete [] h_flagList;
		}
	}

	cudaFree(dlabels);
	cudaFree(dInValues);
	cudaFree(dInColOffset);
	cudaFree(dInRowOffset);
	cudaFree(dOutValues);
	cudaFree(dOutColOffset);
	cudaFree(dOutRowOffset);
/*	
	int tempCount = 0;
	for(int i = 0; i < qsize; i++)
        {
                for(int j = query->inRowOffset[i]; j < query->inRowOffset[i+1]; j ++)
		{
			printf("here is the egde can of %d-->%d:\n",query->inColOffset[j],i);
			for(int k = 0; k < edgeCansSize[tempCount]; k ++)
				printf("cans for %d-->%d  :  %d-->%d\n",query->inColOffset[j],i,edgeCans[tempCount][2*k+1],edgeCans[tempCount][2*k]);
			tempCount++;
		}
	}
*/			
	//find final ans	
	printf("start find final ans\n");
	int edgeCansOff[query->edgeSize+1];
	memset(edgeCansOff,0,sizeof(int)*(query->edgeSize+1));
	edgeCansOff[0] = 0;
	for(int i = 1; i <= query->edgeSize; i ++)
		edgeCansOff[i] = edgeCansOff[i-1] + edgeCansSize[i-1];
	int* d_edgeCansOff;
	int* d_edgeCans;
	cudaMalloc(&d_edgeCansOff, sizeof(int)*(query->edgeSize+1));
	cudaMalloc(&d_edgeCans,sizeof(int)*edgeCansOff[query->edgeSize]*2);
	cudaMemcpy((void *)d_edgeCansOff, (void *)edgeCansOff, sizeof(int)*(query->edgeSize+1), cudaMemcpyHostToDevice);
	int totalThreadNum = 1;
	for(int i = 0; i < query->edgeSize; i ++){
		totalThreadNum *= edgeCansSize[i];
		printf("the edgeCansSize %d is %d\n",i,edgeCansSize[i]);
		cudaMemcpy((void *)(d_edgeCans + edgeCansOff[i]*2),(void *)edgeCans[i], sizeof(int)*edgeCansSize[i]*2, cudaMemcpyHostToDevice);
	}
//	for(int i = 0; i <= query->edgeSize; i ++)
//		printf("%d ",edgeCansOff[i]);
//	printf("\n");
	if(totalThreadNum == 0)
		return ;
	printf("the total thread num is %d\n",totalThreadNum);
	int* flagList = new int[totalThreadNum];
	int* d_ansArea;
	memset(flagList,0,sizeof(int)*totalThreadNum);
	//cudaMemcpy((void *)d_flagList, (void *)flagList, sizeof(int)*totalThreadNum, cudaMemcpyHostToDevice);
	int* d_temMatchResult;
	cudaMalloc(&d_temMatchResult, sizeof(int)*MAXTOTALTHREAD*qsize);
	for(int i = 0; i < (totalThreadNum+MAXTOTALTHREAD-1)/MAXTOTALTHREAD; i ++)
	{
		//if(i < 1000 || i > 1200)	continue;
		int startPos = i*MAXTOTALTHREAD;
		int curThreadNum = totalThreadNum - startPos;
		if(curThreadNum > MAXTOTALTHREAD)
				curThreadNum = MAXTOTALTHREAD;	
		int* d_flagList;
		cudaMalloc(&d_flagList, sizeof(int)*curThreadNum);
		cudaMemcpy((void *)d_flagList, (void *)(flagList + startPos),sizeof(int)*curThreadNum, cudaMemcpyHostToDevice);
		cudaMemset((void *)d_temMatchResult,-1, sizeof(int)*MAXTOTALTHREAD*qsize);
	//	printf("the curThreadNum is %d\n",curThreadNum);
		edgeMerge1<<<(curThreadNum+MAXTHREADPERBLOCK-1)/MAXTHREADPERBLOCK,MAXTHREADPERBLOCK>>>(d_edgeCansOff,
																								d_edgeCans,
																								d_flagList,
																								curThreadNum,
																								qsize,
																								d_m,
																								startPos,
																								query->edgeSize,
																								d_temMatchResult);
		//cudaDeviceSynchronize();
		cudaMemcpy((void *)(flagList + startPos), (void *)d_flagList, sizeof(int)*curThreadNum, cudaMemcpyDeviceToHost);
		cudaFree(d_flagList);
		//cudaDeviceSynchronize();
	}
	cudaFree(d_temMatchResult);

	int effectiveSize = 0;
	for(int i = 0; i < totalThreadNum; i ++)
		if(flagList[i] != 0)
		{
			flagList[effectiveSize++] = i;
		}
	printf("the effective size is %d\n",effectiveSize);
	if(effectiveSize == 0)
		return;
	int* d_flagList;
	cudaMalloc(&d_flagList,sizeof(int)*effectiveSize);
	cudaMemcpy((void *)d_flagList, (void *)flagList, sizeof(int)*effectiveSize, cudaMemcpyHostToDevice);
	cudaMalloc(&d_ansArea,sizeof(int)*qsize*effectiveSize);
	delete [] flagList;
	int* h_ansArea = new int[effectiveSize*qsize];


		edgeMerge2<<<(effectiveSize+MAXTHREADPERBLOCK-1)/MAXTHREADPERBLOCK,MAXTHREADPERBLOCK>>>(d_edgeCansOff,
																								d_edgeCans,
																								d_flagList,
																								effectiveSize,
																								d_ansArea,
																								qsize,
																								d_m,
																								query->edgeSize);
		
		cudaMemcpy((void *)h_ansArea, (void *) d_ansArea, sizeof(int)*qsize*effectiveSize, cudaMemcpyDeviceToHost);


	//PrintAns
	for(int i = 0; i < effectiveSize; i ++)
	{
		int* startPos = h_ansArea + qsize*i;
		for(int j = 0; j < qsize; j ++)
		{
			printf("(%d, %d) ", j, startPos[j]);
		}
		printf("\n");
	}

	cudaFree(d_flagList);
	cudaFree(d_edgeCans);
	cudaFree(d_edgeCansOff);
	cudaFree(d_ansArea);

	delete [] h_ansArea;

	cudaFree(qlabels);
	cudaFree(qInRowOffset);
	cudaFree(qInColOffset);
	cudaFree(qInValues);
	cudaFree(qOutValues);
	cudaFree(qOutColOffset);
	cudaFree(qOutRowOffset);


}
























