/*=============================================================================
# Filename: Graph.h
# Author: Bookug Lobert 
# Mail: 1181955272@qq.com
# Last Modified: 2016-10-24 23:00
# Description: 
=============================================================================*/

#ifndef _GRAPH_GRAPH_H
#define _GRAPH_GRAPH_H

#include "../util/Util.h"


class Neighbor
{
public:
	VID vid;
	LABEL elb;
	Neighbor()
	{
		vid = -1;
		elb = -1;
	}
	Neighbor(long _vid, long _elb)
	{
		vid = _vid;
		elb = _elb;
	}
};

class Vertex
{
public:
	//VID id;
	LABEL label;
	//NOTICE:VID and EID is just used in this single graph
	std::vector<Neighbor> in;
	std::vector<Neighbor> out;
	Vertex()
	{
		label = -1;
	}
	Vertex(LABEL lb):label(lb)
	{
	}
};

class Graph
{
public:
	std::vector<Vertex> vertices;
	long* labels;

	long nodeSize;
	long edgeSize;
	long nodeLabelNum;
	long edgeLabelNum;

	long totalInNum;
	long totalOutNum;

	long* inRowOffset;//行偏移
	long* inColOffset;//列偏移，也就是邻接点的vid
	long* inValues;//values，也就是边的labels

	long* outRowOffset;
	long* outColOffset;
	long* outValues;

	long* reverseLabel;
	long* reverseValues;

	long* diffLabelNum;//in fact diffLabelSum

	Graph() { }
	~Graph() {
		delete [] labels;
	    delete [] inRowOffset;//行偏移
		delete [] inColOffset;//列偏移，也就是邻接点的vid
		delete [] inValues;//values，也就是边的labels

		delete [] outRowOffset;
		delete [] outColOffset;
		delete [] outValues;

		delete [] reverseValues;
		delete [] reverseLabel;

		delete [] diffLabelNum;
	}
	long vSize() const
	{
		return vertices.size();
	}
	void addVertex(LABEL _vlb);
	void addEdge(VID _from, VID _to, LABEL _elb);
	void buildCSR();
	void buildReverse();
};

#endif

