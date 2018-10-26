/*=============================================================================
# Filename: Match.h
# Author: Bookug Lobert 
# Mail: 1181955272@qq.com
# Last Modified: 2016-10-24 22:55
# Description: find all subgraph-graph mappings between query graph and data graph
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

	int qsize, dsize;
	
	Graph* query;
	Graph* data;

	int** nodeCans;
        int* nodeCansSize;
        int** edgeCans;
        int* edgeCansSize;

	
//WARNING: all the above variable are not avaliable on GPU!!!!!
	
	int* qlabels;
	int qTotalInNum;
	int qTotalOutNum;

	int* qInRowOffset;
	int* qInColOffset;
	int* qInValues;

	int* qOutRowOffset;
	int* qOutColOffset;
	int* qOutValues;


	int* dlabels;
	int dTotalInNum;
	int dTotalOutNum;
	
	int* dInRowOffset;
	int* dInColOffset;
	int* dInValues;

	int* dOutRowOffset;
	int* dOutColOffset;
	int* dOutValues;
		
};

class Match
{
public:
	Match(Graph* _query, Graph* _data);
	void match(IO& io);
	~Match();

private:

	int qsize, dsize;
	Graph* query;
	Graph* data;

	int** nodeCans;
	int* nodeCansSize;
	int** edgeCans;
	int* edgeCansSize; 

//structures for d_Match
	int* qlabels;
	int qTotalInNum;
	int qTotalOutNum;

	int* qInRowOffset;
	int* qInColOffset;
	int* qInValues;

	int* qOutRowOffset;
	int* qOutColOffset;
	int* qOutValues;


	int* dlabels;
	int dTotalInNum;
	int dTotalOutNum;
	
	int* dInRowOffset;
	int* dInColOffset;
	int* dInValues;

	int* dOutRowOffset;
	int* dOutColOffset;
	int* dOutValues;

};


#endif

