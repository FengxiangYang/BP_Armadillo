#pragma once
#include <iostream>
#include <armadillo>
#include "vector"
using namespace std;
using namespace arma;
typedef struct InputLayer//输入层结构体定义
{
	mat SampleData, Value;//输入的样本向量以及训练数据矩阵
}Input;
typedef struct HiddenLayer//隐层结构体定义
{
	mat w, delta, Dw, Value, Response;//定义权重矩阵，残差矩阵，权重梯度矩阵，输入值矩阵，sig激活值矩阵
	double bias, Db;//定义偏执
}Hidden;
typedef struct Outlayer//输出层结构体定义
{
	mat w, Dw,Value,delta,Response,Label,LabelData;//定义权重矩阵，残差矩阵，权重梯度矩阵，输入值矩阵，sig激活值矩阵以及标签
}Out;
class BPnet
{
public:
	BPnet(mat Data,mat Labels);//构造函数
	BPnet(BPnet &net);//拷贝构造函数
	mat sigmoid(mat vec);//求取sigmoid相应
	void forward();//误差前向传播
	void backproporgation();//误差反向传播
	void train();//训练函数
	void predict(mat Data);//测试函数
	~BPnet();//析构
public:
	int  HiddenLayerNum;//隐层层数
	int InputNeuronNum, OutNeuronNum;//输入与输出神经元数
	vector<int>  HiddenNeuronNum;//隐层神经元每一层的数目
	double err, alpha, lambda, eplision;//代价函数,学习速率，正则项系数
	Out *OutLayer;//输出层
	vector<Hidden*>HiddenLayer;//隐层指针vector,使用vector方便动态分配
	Input *InputLayer;//输入层
};

