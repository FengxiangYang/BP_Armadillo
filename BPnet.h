#pragma once
#include <iostream>
#include <armadillo>
#include "vector"
using namespace std;
using namespace arma;
typedef struct InputLayer//�����ṹ�嶨��
{
	mat SampleData, Value;//��������������Լ�ѵ�����ݾ���
}Input;
typedef struct HiddenLayer//����ṹ�嶨��
{
	mat w, delta, Dw, Value, Response;//����Ȩ�ؾ��󣬲в����Ȩ���ݶȾ�������ֵ����sig����ֵ����
	double bias, Db;//����ƫִ
}Hidden;
typedef struct Outlayer//�����ṹ�嶨��
{
	mat w, Dw,Value,delta,Response,Label,LabelData;//����Ȩ�ؾ��󣬲в����Ȩ���ݶȾ�������ֵ����sig����ֵ�����Լ���ǩ
}Out;
class BPnet
{
public:
	BPnet(mat Data,mat Labels);//���캯��
	BPnet(BPnet &net);//�������캯��
	mat sigmoid(mat vec);//��ȡsigmoid��Ӧ
	void forward();//���ǰ�򴫲�
	void backproporgation();//���򴫲�
	void train();//ѵ������
	void predict(mat Data);//���Ժ���
	~BPnet();//����
public:
	int  HiddenLayerNum;//�������
	int InputNeuronNum, OutNeuronNum;//�����������Ԫ��
	vector<int>  HiddenNeuronNum;//������Ԫÿһ�����Ŀ
	double err, alpha, lambda, eplision;//���ۺ���,ѧϰ���ʣ�������ϵ��
	Out *OutLayer;//�����
	vector<Hidden*>HiddenLayer;//����ָ��vector,ʹ��vector���㶯̬����
	Input *InputLayer;//�����
};

