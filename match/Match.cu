/*=============================================================================
# Filename: Match.cu
# Author: bookug 
# Mail: bookug@qq.com
# Last Modified: 2018-10-29 20:52
# Description: 
=============================================================================*/

#include "Match.cuh"
#include <cuda_runtime.h>

#define MAXTHREADPERBLOCK 512
#define MAXTOTALTHREAD 1000000
/*#define BITS sizeof(long)*8*/
#define BITS 1

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

    cudaMalloc(&qlabels, sizeof(long)*qsize);
    cudaMalloc(&qInRowOffset, sizeof(long)*(qsize+1));
    cudaMalloc(&qOutRowOffset, sizeof(long)*(1+qsize));
    cudaMalloc(&qInValues,sizeof(long)*qTotalInNum);
    cudaMalloc(&qInColOffset,sizeof(long)*qTotalInNum);
    cudaMalloc(&qOutValues,sizeof(long)*qTotalOutNum);
    cudaMalloc(&qOutColOffset,sizeof(long)*qTotalOutNum);

    cudaMalloc(&dlabels, sizeof(long)*dsize);
    cudaMalloc(&dInRowOffset, sizeof(long)*(dsize+1));
    cudaMalloc(&dOutRowOffset, sizeof(long)*(1+dsize));
    cudaMalloc(&dInValues,sizeof(long)*dTotalInNum);
    cudaMalloc(&dInColOffset,sizeof(long)*dTotalInNum);
    cudaMalloc(&dOutValues,sizeof(long)*dTotalOutNum);
    cudaMalloc(&dOutColOffset,sizeof(long)*dTotalOutNum);

    //	printf("malloc in build match all done\n");
    //	printf("the qsize is %ld,and the query labels is \n",qsize);
    //	for(long i = 0; i < qsize; i++)
    //		printf("the qlabel %ld is %ld\n",i,_query->labels[i]);	
    cudaMemcpy((void *)qlabels, (void *)_query->labels, sizeof(long)*qsize, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)qInRowOffset, (void *)_query->inRowOffset, sizeof(long)*(qsize+1), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)qOutRowOffset, (void *)_query->outRowOffset, sizeof(long)*(qsize+1), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)qInColOffset, (void *)_query->inColOffset, sizeof(long)*qTotalInNum, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)qOutColOffset, (void *)_query->outColOffset, sizeof(long)*qTotalOutNum, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)qInValues, (void *)_query->inValues, sizeof(long)*qTotalInNum, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)qOutValues, (void *)_query->outValues, sizeof(long)*qTotalOutNum, cudaMemcpyHostToDevice);

    cudaMemcpy((void *)dlabels, (void *)_data->labels, sizeof(long)*dsize, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)dInRowOffset, (void *)_data->inRowOffset, sizeof(long)*(dsize+1), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)dOutRowOffset, (void *)_data->outRowOffset, sizeof(long)*(dsize+1), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)dInColOffset, (void *)_data->inColOffset, sizeof(long)*dTotalInNum, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)dOutColOffset, (void *)_data->outColOffset, sizeof(long)*dTotalOutNum, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)dInValues, (void *)_data->inValues, sizeof(long)*dTotalInNum, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)dOutValues, (void *)_data->outValues, sizeof(long)*dTotalOutNum, cudaMemcpyHostToDevice);
    //TODO:
    //cudaDeviceSynchronize();
    printf("move csr to GPU done\n");
    //cuda init
    long t_0,t_1;
    t_0 = Util::get_cur_time();
    long* warmup = NULL;
    cudaMalloc(&warmup, sizeof(long));
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
    printf("cuda init done,it uses %ld ms\n",t_1 - t_0);

}

Match::~Match()
{

    for(long i = 0; i < qsize; i ++)
        delete [] nodeCans[i];
    for(long i = 0; i < query->edgeSize; i ++)
        delete [] edgeCans[i];
    delete [] nodeCansSize;
    delete [] edgeCansSize;
    delete [] nodeCans;
    delete [] edgeCans;
}

//Two States of core queues: matching [0,size) not-matching -1
//Four States of in/out queues: matching depth   in_queue depth  out_queue depth other -1
//NOTICE:a vertex maybe appear in in_queue and out_queue at the same time
__global__ void findEgdeCansInDevice1(long label,
        long* toDevice,
        long nodeCansSize1,
        long nodeCansSize2,
        long totalThreadNum,
        long* d_flagList,
        d_Match* d_m)
{
    long idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= totalThreadNum)
        return;
    bool reverse = false;
    long canSize = nodeCansSize2;
    if(nodeCansSize1 < nodeCansSize2){
        reverse = true;
        canSize = nodeCansSize1;
    }
    long* cans1;
    long* cans2;
    if(reverse)
    {
        cans1 = toDevice + nodeCansSize1;
        cans2 = toDevice;
    }else{
        cans1 = toDevice;
        cans2 = toDevice + nodeCansSize1;
    }
    long ansNum = 0;
    long curId = cans1[idx];
    long* adjList;
    long* adjLable;
    long adjListSize;
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
    long i = 0,j =0;
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

__global__ void findEgdeCansInDevice2(long label,
        long* toDevice,
        long nodeCansSize1,
        long nodeCansSize2,
        long totalThreadNum,
        long* d_flagList,
        long* ansArea,
        d_Match* d_m)
{
    long idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= totalThreadNum)
        return;
    bool reverse = false;
    long canSize = nodeCansSize2;
    long cansWritePos = 1;
    if(nodeCansSize1 < nodeCansSize2){
        reverse = true;
        canSize = nodeCansSize1;
        cansWritePos = 0;
    }
    long* cans1;
    long* cans2;
    if(reverse)
    {
        cans1 = toDevice + nodeCansSize1;
        cans2 = toDevice;
    }else{
        cans1 = toDevice;
        cans2 = toDevice + nodeCansSize1;
    }
    long startPos = d_flagList[idx];
    long curId = cans1[idx];
    long* adjList;
    long* adjLable;
    long adjListSize;
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
    long i = 0,j =0;
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
__global__ void edgeMerge1(long * flagArray,
        long * edgeCans,
        long edgeCansSize,
        long qid1,
        long qid2,
        long totalThreadNum,
        long qsize,
        long * effectBit)
{
    long idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= totalThreadNum)
        return ;
    long edgeId = idx%edgeCansSize;
    long flagId = idx/edgeCansSize;
    long* curFlagId = flagArray + flagId*qsize;
    if(curFlagId[qid1] != -1)
    {
        if(curFlagId[qid1] != edgeCans[2*edgeId])
            return;
    }
    if(curFlagId[qid2] != -1)
    {
        if(curFlagId[qid2] != edgeCans[2*edgeId+1])
            return;
    }
    for(long i = 0; i < qsize; i ++)
        for(long j = i+1; j < qsize; j ++)
        {
            if(curFlagId[i]!= -1 && curFlagId[i] == curFlagId[j])
            {
                return ;      
            }
        }
    for(long i = 0; i < qsize; i++)
    {
        if(i != qid1 && curFlagId[i] == edgeCans[2*edgeId])
            return ;
        if(i != qid2 && curFlagId[i] == edgeCans[2*edgeId+1])
            return ;
    }
    effectBit[idx] = 1;
    return ;
}
__global__ void edgeMerge2(long * flagArray,
        long * edgeCans,
        long edgeCansSize,
        long qid1,
        long qid2,
        long totalThreadNum,
        long qsize,
        long * effectBit,
        long * ansArea)
{
    long realIdx = blockIdx.x*blockDim.x + threadIdx.x;
    if(realIdx >= totalThreadNum)
        return ;
    long idx = effectBit[realIdx];
    long edgeId = idx%edgeCansSize;
    long flagId = idx/edgeCansSize;
    long* curFlagId = flagArray + flagId*qsize;
    memcpy(ansArea+realIdx*qsize,curFlagId,sizeof(long)*qsize);
    curFlagId = ansArea+realIdx*qsize;
    curFlagId[qid1] = edgeCans[2*edgeId];
    curFlagId[qid2] = edgeCans[2*edgeId+1];
    return ;
}
/*
   __global__ void edgeMerge1(long* d_edgeCansOff,
   long* d_edgeCans,
   long* d_flagList,
   long totalThreadNum,
   long qsize,
   d_Match* d_m,
   long startPos,
   long edgeNum,
   long * d_matchResult)
   {
   long idx = blockIdx.x*blockDim.x + threadIdx.x;
   if(idx >= totalThreadNum)
   return;
//	if(idx%MAXTHREADPERBLOCK== 0)
//		printf("here is idx 0 from %ld\n",blockIdx.x);
long* aMatch = d_matchResult + idx*qsize;
idx = idx + startPos;
long mul = 1;
long edgeCount = edgeNum - 1;
bool prune = false;
for(long i = qsize-1; i >= 0; i--)
{

for(long j = d_m->qInRowOffset[i+1]-1; j >= d_m->qInRowOffset[i]; j --)
{
long n1 = d_edgeCans[(d_edgeCansOff[edgeCount] + (idx/mul)%(d_edgeCansOff[edgeCount+1] - d_edgeCansOff[edgeCount]))*2];
long n2 = d_edgeCans[(d_edgeCansOff[edgeCount] + (idx/mul)%(d_edgeCansOff[edgeCount+1] - d_edgeCansOff[edgeCount]))*2+1];
//if(edgeCount == 0)
//	printf("the edgeCount is 0, the cans is %ld, %ld\n",n1,n2);
mul *= (d_edgeCansOff[edgeCount+1] - d_edgeCansOff[edgeCount]);
edgeCount --;
long q2 = d_m->qInColOffset[j];
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
for(long i = 0; i < qsize; i ++)
for(long j = i+1; j < qsize; j ++)
{
if(aMatch[i] == aMatch[j])
{
prune = true;
}
}
if(prune)
{
if(
aMatch[0] == 4280
&& aMatch[1] == 4279
aMatch[2] == 4249
&& aMatch[3] == 4248
&& aMatch[4] == 1075
&& aMatch[5] == 1072
&& aMatch[6] == 1071
)
printf("%ld %ld %ld %ld %ld %ld %ld\n",aMatch[0],aMatch[1],aMatch[2],aMatch[3],aMatch[4],aMatch[5],aMatch[6]); 
return ;
}
d_flagList[idx-startPos] = 1;
//printf("here i get an answer!\n");
return ;

}
__global__ void edgeMerge2(long* d_edgeCansOff,
        long* d_edgeCans,
        long* d_flagList,
        long effectiveSize,
        long* d_ansArea,
        long qsize,
        d_Match* d_m,
        long edgeNum)
{
    long real_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(real_idx >= effectiveSize)
        return;

    long idx = d_flagList[real_idx];
    long *writePos = d_ansArea + real_idx*qsize;
    long mul = 1;
    long edgeCount = edgeNum - 1;
    for(long i = qsize-1; i >= 0; i--)
    {
        for(long j = d_m->qInRowOffset[i+1]-1; j >= d_m->qInRowOffset[i]; j --)
        {
            long n1 = d_edgeCans[(d_edgeCansOff[edgeCount] + (idx/mul)%(d_edgeCansOff[edgeCount+1] - d_edgeCansOff[edgeCount]))*2];
            long n2 = d_edgeCans[(d_edgeCansOff[edgeCount] + (idx/mul)%(d_edgeCansOff[edgeCount+1] - d_edgeCansOff[edgeCount]))*2+1];

            mul *= (d_edgeCansOff[edgeCount+1] - d_edgeCansOff[edgeCount]);
            edgeCount --;

            writePos[i] = n1;

            writePos[d_m->qInColOffset[j]] = n2;		
        }
    }
    return ;
}*/
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
    nodeCans = new long*[qsize];
    nodeCansSize  = new long[qsize];
    /*	for(long i = 0; i <= data->nodeLabelNum; i ++)
        printf("%ld\n",data->reverseLabel[i]);
        printf(" \n\n");
        for(long i = 0; i < data->reverseLabel[data->nodeLabelNum];i ++)
        printf("%ld\n",data->reverseValues[i]);


        for(long i = 0; i <= query->nodeLabelNum; i ++)
        printf("%ld\n",query->reverseLabel[i]);
        printf(" \n\n");
        for(long i = 0; i < query->reverseLabel[query->nodeLabelNum];i ++)
        printf("%ld\n",query->reverseValues[i]);*/

    for(long i = 0; i < qsize; i ++)
    {
        long targerLabel = query->vertices[i].label - 1;
        long targetDegree = query->vertices[i].in.size() + query->vertices[i].out.size();
        long* tempCans = new long [data->reverseLabel[targerLabel+1] - data->reverseLabel[targerLabel]];
        //	printf("the can size is %ld, targetLabel is %ld,degree is %ld\n",
        //		data->reverseLabel[targerLabel+1] - data->reverseLabel[targerLabel],targerLabel,targetDegree);
        long cansPos = 0;
        for(long j = data->reverseLabel[targerLabel]; j < data->reverseLabel[targerLabel+1]; j ++)
        {
            long curId = data->reverseValues[j];
            //		printf("j is %ld,curId is %ld, degree is %ld\n",j,curId,data->vertices[curId].in.size() + data->vertices[curId].out.size());
            if(data->vertices[curId].in.size() >= query->vertices[i].in.size()&&
                    data->vertices[curId].out.size() >= query->vertices[i].out.size()){
                tempCans[cansPos++] = curId;
            }
        }
        nodeCans[i] = new long[cansPos];
        nodeCansSize[i] = cansPos;
        memcpy(nodeCans[i],tempCans,sizeof(long)*cansPos);
        delete [] tempCans;
    }
    /*for(long i = 0; i < qsize; i++)
      {
      printf("here is cans for node %ld\n",i);
      for(long j = 0; j < nodeCansSize[i];j ++)
      printf("%ld\n",nodeCans[i][j]);
      }*/	
    //find cans for edges
    printf("start find cans edges\n");
    edgeCans = new long*[query->edgeSize];
    edgeCansSize = new long[query->edgeSize];
    //TODO:here I use inEdge to do all things
    long edgeCount = 0;
    for(long i = 0; i < qsize; i++)
    {
        for(long j = query->inRowOffset[i]; j < query->inRowOffset[i+1]; j ++)
        {
            long targerLabel = query->inValues[j];
            long anotherNode = query->inColOffset[j];
            long* toDevice;
            long totalThreadNum = nodeCansSize[i];
            if(totalThreadNum < nodeCansSize[anotherNode])
                totalThreadNum = nodeCansSize[anotherNode];

            if(totalThreadNum == 0)
                return ;
            long* d_flagList;
            long* h_flagList = new long[totalThreadNum+1];
            memset(h_flagList,0,sizeof(long)*(1+totalThreadNum));
            cudaMalloc(&d_flagList, sizeof(long)*totalThreadNum);
            cudaMemcpy((void *)d_flagList, (void *)(h_flagList+1), sizeof(long)*totalThreadNum, cudaMemcpyHostToDevice);
            cudaMalloc(&toDevice, sizeof(long)*(nodeCansSize[i]+nodeCansSize[anotherNode]));
            cudaMemcpy((void *)toDevice,nodeCans[i],sizeof(long)*nodeCansSize[i],cudaMemcpyHostToDevice);
            cudaMemcpy((void *)(toDevice + nodeCansSize[i]),nodeCans[anotherNode],sizeof(long)*nodeCansSize[anotherNode], cudaMemcpyHostToDevice);
            findEgdeCansInDevice1<<<(totalThreadNum+MAXTHREADPERBLOCK-1)/MAXTHREADPERBLOCK,MAXTHREADPERBLOCK>>>(targerLabel,
                    toDevice,
                    nodeCansSize[i],
                    nodeCansSize[anotherNode],
                    totalThreadNum,
                    d_flagList,
                    d_m);
            cudaMemcpy((void *)(h_flagList+1),(void *)d_flagList,sizeof(long)*totalThreadNum,cudaMemcpyDeviceToHost);
            cudaFree(d_flagList);

            //		printf("here is the h_flagList\n");
            //		for(long k = 0; k <= totalThreadNum; k++)
            //			printf("h_flagList[%ld] is %ld\n",k,h_flagList[k]);

            for(long k = 1; k <= totalThreadNum; k ++)
            {
                h_flagList[k] = h_flagList[k] + h_flagList[k-1]; 
            }


            long* d_ansArea;
            cudaMalloc(&d_ansArea, sizeof(long)*2*h_flagList[totalThreadNum]);
            cudaMalloc(&d_flagList,sizeof(long)*(totalThreadNum+1));
            cudaMemcpy((void *)d_flagList, (void *)h_flagList, sizeof(long)*(totalThreadNum+1), cudaMemcpyHostToDevice);
            findEgdeCansInDevice2<<<(totalThreadNum+MAXTHREADPERBLOCK-1)/MAXTHREADPERBLOCK,MAXTHREADPERBLOCK>>>(targerLabel,
                    toDevice,
                    nodeCansSize[i],
                    nodeCansSize[anotherNode],
                    totalThreadNum,
                    d_flagList,
                    d_ansArea,
                    d_m);
            edgeCans[edgeCount] = new long[h_flagList[totalThreadNum]*2];
            cudaMemcpy((void *)edgeCans[edgeCount],(void *)d_ansArea,sizeof(long)*2*h_flagList[totalThreadNum],cudaMemcpyDeviceToHost);
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
        long tempCount = 0;
        for(long i = 0; i < qsize; i++)
        {
        for(long j = query->inRowOffset[i]; j < query->inRowOffset[i+1]; j ++)
        {
        printf("here is the egde can of %ld-->%ld:\n",query->inColOffset[j],i);
        for(long k = 0; k < edgeCansSize[tempCount]; k ++)
        printf("cans for %ld-->%ld  :  %ld-->%ld\n",query->inColOffset[j],i,edgeCans[tempCount][2*k+1],edgeCans[tempCount][2*k]);
        tempCount++;
        }
        }
     */			
    //find final ans	
    edgeCount = 0;
    long midResultNum = 1;
    long* d_flagArray;
    long* d_ansArea;
    long effectiveSize = 0;
    for(long i = 0; i < qsize; i++)
    {
        for(long j = query->inRowOffset[i]; j < query->inRowOffset[i+1]; j ++)
        {
            if(edgeCount == 0)
            {
                cudaMalloc(&d_flagArray, sizeof(long)*qsize*edgeCansSize[0]);
                cudaMemset(d_flagArray, -1, sizeof(long)*qsize*edgeCansSize[0]);
            }
            long totalThreadNum = midResultNum * edgeCansSize[edgeCount];
            printf("the totalNum is %ld\n",totalThreadNum);
            /*long* effectBit =  new long[totalThreadNum];*/
            long* effectBit =  new long[totalThreadNum/BITS];
            long* d_effectBit;
            long* d_edgeCans;
            cudaMalloc(&d_edgeCans, sizeof(long)*edgeCansSize[edgeCount]*2);
            cudaMemcpy((void *)d_edgeCans, (void *)edgeCans[edgeCount], sizeof(long)*edgeCansSize[edgeCount]*2, cudaMemcpyHostToDevice);
            memset(effectBit,0,sizeof(long)*totalThreadNum/BITS);
            cudaMalloc(&d_effectBit, sizeof(long)*totalThreadNum/BITS);
            cudaMemcpy((void *)d_effectBit, (void *)effectBit, sizeof(long)*totalThreadNum/BITS, cudaMemcpyHostToDevice);

            edgeMerge1<<<(totalThreadNum+MAXTHREADPERBLOCK-1)/MAXTHREADPERBLOCK,MAXTHREADPERBLOCK>>>(d_flagArray,
                    d_edgeCans,
                    edgeCansSize[edgeCount],
                    i,
                    query->inColOffset[j],
                    totalThreadNum,
                    qsize,
                    d_effectBit);

            cudaMemcpy((void *)effectBit, (void *)d_effectBit, sizeof(long)*totalThreadNum/BITS,cudaMemcpyDeviceToHost);
            effectiveSize = 0;
            //BETTER: finish this using scatter operation on GPU
            for(long k = 0; k < totalThreadNum; k ++){
                if(effectBit[k] != 0){
                    effectBit[effectiveSize++] = k;
                }
            }

            cudaFree(d_effectBit);
            cudaMalloc(&d_ansArea, sizeof(long)*effectiveSize*qsize);
            cudaMalloc(&d_effectBit, sizeof(long)*effectiveSize);
            cudaMemcpy((void *)d_effectBit, (void *)effectBit, sizeof(long)*effectiveSize, cudaMemcpyHostToDevice);
            edgeMerge2<<<(effectiveSize+MAXTHREADPERBLOCK-1)/MAXTHREADPERBLOCK,MAXTHREADPERBLOCK>>>(d_flagArray,
                    d_edgeCans,
                    edgeCansSize[edgeCount],
                    i,
                    query->inColOffset[j],
                    effectiveSize,
                    qsize,
                    d_effectBit,
                    d_ansArea);
            edgeCount ++;
            printf("after edge %ld, the effective size is %ld\n", edgeCount, effectiveSize);
            midResultNum = effectiveSize;
            cudaFree(d_flagArray);
            cudaFree(d_effectBit);
            cudaFree(d_edgeCans);
            delete [] effectBit;
            d_flagArray = d_ansArea;

        }
    }

    cout<<"to output result: "<<effectiveSize<<endl;
    long* h_ansArea = new long [effectiveSize*qsize];
    cudaMemcpy((void *)h_ansArea, (void *)d_ansArea, sizeof(long)*effectiveSize*qsize, cudaMemcpyDeviceToHost);
    cudaFree(d_ansArea);
    for(long i = 0; i < effectiveSize; i ++)
    {
        long* tmpPtr = h_ansArea + qsize*i;
        io.output(tmpPtr, qsize);
    }
    delete [] h_ansArea;
    /*
       printf("start find final ans\n");
       long edgeCansOff[query->edgeSize+1];
       memset(edgeCansOff,0,sizeof(long)*(query->edgeSize+1));
       edgeCansOff[0] = 0;
       for(long i = 1; i <= query->edgeSize; i ++)
       edgeCansOff[i] = edgeCansOff[i-1] + edgeCansSize[i-1];
       long* d_edgeCansOff;
       long* d_edgeCans;
       cudaMalloc(&d_edgeCansOff, sizeof(long)*(query->edgeSize+1));
       cudaMalloc(&d_edgeCans,sizeof(long)*edgeCansOff[query->edgeSize]*2);
       cudaMemcpy((void *)d_edgeCansOff, (void *)edgeCansOff, sizeof(long)*(query->edgeSize+1), cudaMemcpyHostToDevice);
       long totalThreadNum = 1;
       for(long i = 0; i < query->edgeSize; i ++){
       totalThreadNum *= edgeCansSize[i];
       printf("the edgeCansSize %ld is %ld\n",i,edgeCansSize[i]);
       cudaMemcpy((void *)(d_edgeCans + edgeCansOff[i]*2),(void *)edgeCans[i], sizeof(long)*edgeCansSize[i]*2, cudaMemcpyHostToDevice);
       }
    //	for(long i = 0; i <= query->edgeSize; i ++)
    //		printf("%ld ",edgeCansOff[i]);
    //	printf("\n");
    if(totalThreadNum == 0)
    return ;
    printf("the total thread num is %ld\n",totalThreadNum);
    long* flagList = new long[totalThreadNum];
    long* d_ansArea;
    memset(flagList,0,sizeof(long)*totalThreadNum);
    //cudaMemcpy((void *)d_flagList, (void *)flagList, sizeof(long)*totalThreadNum, cudaMemcpyHostToDevice);
    long* d_temMatchResult;
    cudaMalloc(&d_temMatchResult, sizeof(long)*MAXTOTALTHREAD*qsize);
    for(long i = 0; i < (totalThreadNum+MAXTOTALTHREAD-1)/MAXTOTALTHREAD; i ++)
    {
    //if(i < 1000 || i > 1200)	continue;
    long startPos = i*MAXTOTALTHREAD;
    long curThreadNum = totalThreadNum - startPos;
    if(curThreadNum > MAXTOTALTHREAD)
    curThreadNum = MAXTOTALTHREAD;	
    long* d_flagList;
    cudaMalloc(&d_flagList, sizeof(long)*curThreadNum);
    cudaMemcpy((void *)d_flagList, (void *)(flagList + startPos),sizeof(long)*curThreadNum, cudaMemcpyHostToDevice);
    cudaMemset((void *)d_temMatchResult,-1, sizeof(long)*MAXTOTALTHREAD*qsize);
    //	printf("the curThreadNum is %ld\n",curThreadNum);
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
    cudaMemcpy((void *)(flagList + startPos), (void *)d_flagList, sizeof(long)*curThreadNum, cudaMemcpyDeviceToHost);
    cudaFree(d_flagList);
    //cudaDeviceSynchronize();
    }
    cudaFree(d_temMatchResult);

    long effectiveSize = 0;
    for(long i = 0; i < totalThreadNum; i ++)
    if(flagList[i] != 0)
    {
    flagList[effectiveSize++] = i;
    }
    printf("the effective size is %ld\n",effectiveSize);
    if(effectiveSize == 0)
    return;
    long* d_flagList;
    cudaMalloc(&d_flagList,sizeof(long)*effectiveSize);
    cudaMemcpy((void *)d_flagList, (void *)flagList, sizeof(long)*effectiveSize, cudaMemcpyHostToDevice);
    cudaMalloc(&d_ansArea,sizeof(long)*qsize*effectiveSize);
    delete [] flagList;
    long* h_ansArea = new long[effectiveSize*qsize];


    edgeMerge2<<<(effectiveSize+MAXTHREADPERBLOCK-1)/MAXTHREADPERBLOCK,MAXTHREADPERBLOCK>>>(d_edgeCansOff,
            d_edgeCans,
            d_flagList,
            effectiveSize,
            d_ansArea,
            qsize,
            d_m,
            query->edgeSize);

    cudaMemcpy((void *)h_ansArea, (void *) d_ansArea, sizeof(long)*qsize*effectiveSize, cudaMemcpyDeviceToHost);


    //PrlongAns
    for(long i = 0; i < effectiveSize; i ++)
    {
        long* startPos = h_ansArea + qsize*i;
        for(long j = 0; j < qsize; j ++)
        {
            printf("(%ld, %ld) ", j, startPos[j]);
        }
        printf("\n");
    }

    cudaFree(d_flagList);
    cudaFree(d_edgeCans);
    cudaFree(d_edgeCansOff);
    cudaFree(d_ansArea);

    delete [] h_ansArea;
    */
        cudaFree(qlabels);
    cudaFree(qInRowOffset);
    cudaFree(qInColOffset);
    cudaFree(qInValues);
    cudaFree(qOutValues);
    cudaFree(qOutColOffset);
    cudaFree(qOutRowOffset);
}

