/*=============================================================================
# Filename: Match.cuh
# Author: bookug 
# Mail: bookug@qq.com
# Last Modified: 2018-10-29 20:52
# Description: 
=============================================================================*/


#ifndef _MATCH_MATCH_H
#define _MATCH_MATCH_H

#include "../util/Util.h"
#include "../graph/Graph.h"
#include "../io/IO.h"
#include <cuda_runtime.h>


class d_Match
{
public:

	long qsize, dsize;
	
	Graph* query;
	Graph* data;

	long** nodeCans;
        long* nodeCansSize;
        long** edgeCans;
        long* edgeCansSize;

	
//WARNING: all the above variable are not avaliable on GPU!!!!!
	
	long* qlabels;
	long qTotalInNum;
	long qTotalOutNum;

	long* qInRowOffset;
	long* qInColOffset;
	long* qInValues;

	long* qOutRowOffset;
	long* qOutColOffset;
	long* qOutValues;


	long* dlabels;
	long dTotalInNum;
	long dTotalOutNum;
	
	long* dInRowOffset;
	long* dInColOffset;
	long* dInValues;

	long* dOutRowOffset;
	long* dOutColOffset;
	long* dOutValues;
		
};

class Match
{
public:
	Match(Graph* _query, Graph* _data);
	void match(IO& io);
	~Match();

private:

	long qsize, dsize;
	Graph* query;
	Graph* data;

	long** nodeCans;
	long* nodeCansSize;
	long** edgeCans;
	long* edgeCansSize; 

//structures for d_Match
	long* qlabels;
	long qTotalInNum;
	long qTotalOutNum;

	long* qInRowOffset;
	long* qInColOffset;
	long* qInValues;

	long* qOutRowOffset;
	long* qOutColOffset;
	long* qOutValues;


	long* dlabels;
	long dTotalInNum;
	long dTotalOutNum;
	
	long* dInRowOffset;
	long* dInColOffset;
	long* dInValues;

	long* dOutRowOffset;
	long* dOutColOffset;
	long* dOutValues;

};


#endif

