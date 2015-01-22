#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include "iostream"
#include <vector>
#include<sys/time.h>
#include <math.h>
#include<string.h>

using namespace std;

double get_time()
{
	struct timeval t;
	gettimeofday(&t, NULL);
	double d = t.tv_sec + (double) t.tv_usec/1000000;
	return d;
}

double d = 0.85;
int vertex, edge;
double normStop = 0.00001;
vector< vector <int> > matrix;
vector<int> row;
vector<int> outdegree;
vector<double> pgrnk;
vector<double> pgrnk_old;
pthread_barrier_t barr;
pthread_barrier_t barr1;	
pthread_mutex_t mutex;

int* rank;
int threads = 16;
double* norm;
int* partition;
int* row_num;

void* pagerank(void*);

int main(int argc, char *argv[])
{
	FILE *fp;	
	if(argc != 3)
	{
		printf("\nUsage: Enter the name of the file containing the Page-Ranks and number of threads. \n");
		exit(EXIT_FAILURE);
	}
	fp = fopen(argv[1], "r");
	if(fp==NULL)
	{
		printf("Unable to open file\n");
		exit(EXIT_FAILURE);
	}
	char line[512];

	threads = atoi(argv[2]);

	fgets(line, sizeof(line), fp);
	fgets(line, sizeof(line), fp);
	fgets(line, sizeof(line), fp);
	sscanf(line, "# Nodes: %d Edges: %d", &vertex, &edge);
	int jj;
	for(jj=0;jj<=vertex;jj++)
	{
		pgrnk.push_back(1/(double) vertex);
		pgrnk_old.push_back(1/(double) vertex);
		outdegree.push_back(0);
	}

	for(int it=0;it<vertex+1;it++)
		matrix.push_back(row);
	int from, to;
	fgets(line, sizeof(line), fp);
	while(1)
	{
		fgets(line, sizeof(line), fp);		
		sscanf(line, "%d	%d", &from, &to);
		if(feof(fp))
			break;
		outdegree[from] = outdegree[from] + 1;
		matrix[to].push_back(from);
		if(feof(fp))
			break;
	}

	pthread_t thr[threads];
	if(pthread_barrier_init(&barr, NULL, threads))
	{
		printf("Could not create a barrier\n");
		return -1;
	}
	if(pthread_barrier_init(&barr1, NULL, threads))
	{
		printf("Could not create a barrier\n");
		return -1;
	}

	if(pthread_mutex_init(&mutex, NULL))
	{
		printf("Unable to initialize a mutex\n");
		return -1;
	}

	rank = (int*) malloc(sizeof(int)*threads);
	partition = (int*) malloc(sizeof(int)*threads);
	row_num = (int*) malloc(sizeof(int)*threads);
	norm = (double*) malloc(sizeof(double)*threads);
	for(int i=0;i<threads;i++){
		partition[i] = vertex%threads>=i ? ((vertex/threads)+1)*i : (vertex/threads)*i+vertex%threads;
		row_num[i] = vertex%threads>i ? (vertex/threads)+1 : vertex/threads;		
	}
	double time_start = get_time();
	for(int i = 0; i < threads; i++)
	{
		rank[i]=0;
		rank[i] = i;
		if(pthread_create(&thr[i], NULL, &pagerank, (void*)(&rank[i])))
		{
			printf("Could not create thread %d\n", i);
			return -1;
		}
	}

	for(int i = 0; i < threads; ++i)
	{
		if(pthread_join(thr[i], NULL))
		{
			printf("Could not join thread %d\n", i);
			return -1;
		}
	}

	double time_end = get_time();

	printf("The final pageranks are as follows\n");
	for(jj=1;jj<=vertex;jj++)
		printf("%.16f\n", pgrnk[jj]);
	printf("\nThe time taken for the execution is %lf", time_end - time_start);
}


void* pagerank(void* arg)
{	
	int jj=0;
	double temp;
	int rank = *((int *)arg);
	int i=0;
	vector<int>::iterator j;

	int start_idx = partition[rank];
	int rows = row_num[rank];
	int end_idx = start_idx+rows;
	double l_vector[rows+1];
	double ln=1,n=1;
	double diff;
	int rc, rc1;

	while(n > normStop){

		for(i=start_idx+1;i<end_idx+1;i++){
			temp = 0.0;
			vector<int> k= matrix.at(i);
			for(j=(k).begin();j!=(k).end();j++)
			{
				temp += d*(pgrnk_old[*j]/outdegree[*j]);
			}
			l_vector[i-start_idx]=temp + (1-d)/(double) vertex;
		}
		rc = pthread_barrier_wait(&barr);
		if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
		{
			printf("Could not wait on barrier\n");
			exit(-1);
		}

		norm[rank]=0;
		for(i=0;i<rows;i++){
			pgrnk[start_idx+i+1]=l_vector[i+1];
			diff = pgrnk[start_idx+i+1]-pgrnk_old[start_idx+i+1];
			norm[rank] += diff*diff;
			pgrnk_old[start_idx+i+1]=pgrnk[start_idx+i+1];
		}
		rc1 = pthread_barrier_wait(&barr1);
		if(rc1 != 0 && rc1 != PTHREAD_BARRIER_SERIAL_THREAD)
		{
			printf("Could not wait on barrier\n");
			exit(-1);
		}
		ln=0;
		for(i=0;i<threads;i++){
			ln+=norm[i];
		}
		n = sqrt(ln);
	}
	pthread_exit(NULL);
}

