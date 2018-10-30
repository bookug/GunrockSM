/*=============================================================================
# Filename: IO.cpp
# Author: Bookug Lobert 
# Mail: 1181955272@qq.com
# Last Modified: 2016-10-24 22:55
# Description: 
=============================================================================*/

#include "IO.h"

using namespace std;

IO::IO()
{
	this->qfp = NULL;
	this->dfp = NULL;
	this->ofp = NULL;
	this->data_id = -1;
}

IO::IO(string query, string data, string file)
{
	this->data_id = -1;
	this->line = "============================================================";
	qfp = fopen(query.c_str(), "r");
	if(qfp == NULL)
	{
		cerr<<"input open error!"<<endl;
		return;
	}
	dfp = fopen(data.c_str(), "r");
	if(dfp == NULL)
	{
		cerr<<"input open error!"<<endl;
		return;
	}
	ofp = fopen(file.c_str(), "w+");
	if(ofp == NULL)
	{
		cerr<<"output open error!"<<endl;
		return;
	}
}

Graph* 
IO::input(FILE* fp)
{
	char c1, c2;
	long id0, id1, id2, lb;
	bool flag = false;
	Graph* ng = NULL;

	while(true)
	{
		fscanf(fp, "%c", &c1);
		if(c1 == 't')
		{
			if(flag)
			{
				fseek(fp, -1, SEEK_CUR);
				ng->buildCSR();
				ng->buildReverse();
				printf("build csr done\n");
				return ng;
			}
			flag = true;
			fscanf(fp, " %c %ld\n", &c2, &id0);
			if(id0 == -1)
			{
				return NULL;
			}
			else
			{
				ng = new Graph;
				fscanf(fp, "%ld %ld %ld %ld\n", &(ng->nodeSize),&(ng->edgeSize),&(ng->nodeLabelNum), &(ng->edgeLabelNum));
				ng->diffLabelNum = new long [ng->nodeLabelNum];
				memset(ng->diffLabelNum,0,sizeof(long)*(ng->nodeLabelNum));
			}
		}
		else if(c1 == 'v')
		{
			fscanf(fp, " %ld %ld\n", &id1, &lb);
			ng->diffLabelNum[lb-1]++;
			ng->addVertex(lb); 
		}
		else if(c1 == 'e')
		{
			fscanf(fp, " %ld %ld %ld\n", &id1, &id2, &lb);
			//NOTICE:we treat this graph as directed, each edge represents two
			//This may cause too many matchings, if to reduce, only add the first one
			ng->addEdge(id1, id2, lb);
	//		ng->addEdge(id2, id1, lb);
		}
		else 
		{
			cerr<<"ERROR in input() -- invalid char"<<endl;
			return false;
		}
	}
	return NULL;
}

bool 
IO::input(Graph*& data_graph)
{
	data_graph = this->input(this->dfp);
	if(data_graph == NULL)
		return false;
	this->data_id++;
	return true;
}

bool 
IO::input(vector<Graph*>& query_list)
{
	Graph* graph = NULL;
	while(true)
	{
		graph = this->input(qfp);
		if(graph == NULL) //to the end
			break;
		query_list.push_back(graph);
	}

	return true;
}

bool 
IO::output(long qid)
{
	fprintf(ofp, "query graph:%ld    data graph:%ld\n", qid, this->data_id);
	fprintf(ofp, "%s\n", line.c_str());
	return true;
}

bool
IO::output()
{
	fprintf(ofp, "\n\n\n");
	return true;
}

bool 
IO::output(long* m, long size)
{
	for(long i = 0; i < size; ++i)
	{
		fprintf(ofp, "(%ld, %ld) ", i, m[i]);
	}
	fprintf(ofp, "\n");
	return true;
}

void
IO::flush()
{
	fflush(this->ofp);
}

IO::~IO()
{
	fclose(this->qfp);
	this->qfp = NULL;
	fclose(this->dfp);
	this->dfp = NULL;
	fclose(this->ofp);
	this->ofp = NULL;
}

