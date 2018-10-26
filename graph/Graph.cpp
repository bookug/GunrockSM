/*=============================================================================
# Filename: Graph.cpp
# Author: Bookug Lobert 
# Mail: 1181955272@qq.com
# Last Modified: 2016-10-24 23:01
# Description: 
=============================================================================*/

#include "Graph.h"

using namespace std;
bool cmpNeighbor(const Neighbor &a, const Neighbor &b)
{
	return a.vid < b.vid;
}

void 
Graph::addVertex(LABEL _vlb)
{
	this->vertices.push_back(Vertex(_vlb));
}

void 
Graph::addEdge(VID _from, VID _to, LABEL _elb)
{
	this->vertices[_from].out.push_back(Neighbor(_to, _elb));
	this->vertices[_to].in.push_back(Neighbor(_from, _elb));
}
void
Graph::buildCSR()
{
	int verSize = this->vSize();
	labels = new int[verSize];
	for(int i = 0; i < verSize; i ++)
		labels[i] = vertices[i].label;

	inRowOffset = new int[verSize + 1];
	inRowOffset[0] = 0;
	outRowOffset = new int[verSize + 1];
	outRowOffset[0] = 0;

	int inTotalNum = 0;
	int outTotalNum = 0;

	for(int i = 0; i < verSize; i ++)
	{
		inTotalNum += this->vertices[i].in.size();
		outTotalNum += this->vertices[i].out.size();
		inRowOffset[i+1] = inTotalNum;
		outRowOffset[i+1] = outTotalNum;
		sort(this->vertices[i].in.begin(),this->vertices[i].in.end(),cmpNeighbor);
		sort(this->vertices[i].out.begin(),this->vertices[i].out.end(),cmpNeighbor);
	}

	inColOffset = new int [inTotalNum];
	inValues = new int [inTotalNum];

	outColOffset = new int[outTotalNum];
	outValues = new int [outTotalNum];

	int inPos = 0;
	int outPos = 0;
	for(int i = 0; i < this->vSize(); i++)
	{
	//	printf("the %d th node's in size is %d,out size is %d\n",i,this->vertices[i].in.size(),this->vertices[i].out.size());
		for(int j = 0; j < this->vertices[i].in.size(); j ++)
		{
			inColOffset[inPos] = ((this->vertices[i].in)[j]).vid;
			inValues[inPos] = ((this->vertices[i].in)[j]).elb;
			inPos ++;
		}
		for(int j = 0; j < this->vertices[i].out.size(); j ++)
		{
			outColOffset[outPos] = ((this->vertices[i].out)[j]).vid;
			outValues[outPos] = ((this->vertices[i].out)[j]).elb;
			outPos ++;
		}
	}
/*
	for(int i = 0; i < verSize; i++)
	{
		printf("######%d\n", i);
		printf("in edge size is %d:\n",inRowOffset[i+1] - inRowOffset[i]);
		for(int j = inRowOffset[i]; j < inRowOffset[i+1]; j ++)
			printf("%d 		%d\n",inColOffset[j],inValues[j]);
		printf("out edge size is %d:\n",outRowOffset[i+1] - outRowOffset[i]);
		for(int j = outRowOffset[i]; j < outRowOffset[i+1]; j ++)
			printf("%d 		%d\n",outColOffset[j],outValues[j]);
	}*/
	this->totalInNum = inTotalNum;
	this->totalOutNum = outTotalNum;
	return ;

}
void 
Graph::buildReverse()
{
	reverseLabel = new int [nodeLabelNum + 1];
	reverseLabel[0] = 0;
//	printf("diffLabelNum is %d\n",diffLabelNum[0]);
	for(int i = 1; i <= nodeLabelNum; i++)
		reverseLabel[i] = reverseLabel[i-1] + diffLabelNum[i-1];
	reverseValues = new int [reverseLabel[nodeLabelNum]];
	int curLabelPos[nodeLabelNum];
	memcpy(curLabelPos,reverseLabel,sizeof(int)*nodeLabelNum);
	for(int i = 0; i < vertices.size(); i ++)
	{
		int label = vertices[i].label - 1;
		reverseValues[curLabelPos[label]] = i;
		curLabelPos[label] ++;
	}

}

