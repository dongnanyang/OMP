//Time:2020-11-9
//author:   KuangXiang
//locate:   Harbin
//github:   https://github.com/dongnanyang
//参考博客MATLAB程序--》https://blog.csdn.net/jbb0523/article/details/45130793
//使用了 Eigen矩阵库


#include<algorithm>
#include<iostream>
#include<time.h>
#include<Eigen/Dense>
using namespace Eigen;
using namespace std;
void randperm(int*a,int N)
{
	for(int i=0;i<N;++i)
		a[i]=i+1;
	for(int i=0;i<N;++i)
	{
		int j=rand()%(N-1);
		swap(a[i],a[j]);
	}
}
VectorXd CS_OMP(MatrixXd A,VectorXd y,int K)
{
	int M=A.rows();
	int N=A.cols();
	VectorXd threta=VectorXd::Zero(N);     //
	MatrixXd At=MatrixXd::Zero(M,K);
	VectorXd Pos_theta=VectorXd::Zero(K);
	VectorXd theta_ls;
	VectorXd r_n=y;
	int pos=0;
	for(int i=0;i<K;++i)
	{
		VectorXd product=A.transpose()*r_n;
		auto productTmp=product.cwiseAbs();
		productTmp.maxCoeff(&pos);
		Pos_theta(i)=pos;
		for(int j=0;j<M;++j)
			At(j,i)=A(j,pos);
		theta_ls=(At.leftCols(i+1).transpose()*At.leftCols(i+1)).inverse()*At.leftCols(i+1).transpose()*y;
		r_n=y-At.leftCols(i+1)*theta_ls;
	}
	for(int i=0;i<K;++i)
	{
		threta((int)Pos_theta(i))=theta_ls(i);
	}
	cout<<"World!"<<endl;
	return threta;
}

int main(int argc,char*agv[])
{
	int i=0,j=0;
	int M=64,N=256,K=10;
	int *Index_K=new int[N];
	randperm(Index_K,N);
	
	VectorXd X=VectorXd::Zero(N);
	for(i=0;i<K;++i)
		X(Index_K[i])=rand()/(double)(RAND_MAX);
	cout<<"原始信号"<<endl;
	cout<<X.transpose()<<endl;
	MatrixXd Psi=MatrixXd::Identity(N,N);
	MatrixXd Phi=(Eigen::MatrixXd::Random(M,N).array());
	auto A=Phi*Psi;
	//cout<<A.cols()<<endl;
	//cout<<A.rows()<<endl;
	auto y=Phi*X;
	//cout<<y.rows()<<endl;
	VectorXd theta;
	//cout<<"h"<<endl;
	theta = CS_OMP(A,y,K);
	cout<<theta.cols()<<endl;
	MatrixXd x_r = Psi*theta;
	cout<<"恢复信号x_r is"<<endl;
	cout<<x_r.transpose()<<endl;
	
	int miss=0;
	for(i=0;i<N;++i)
		miss=miss+abs(x_r(i)-X(i));
	cout<<"损失为 is"<<miss<<endl;
	
}