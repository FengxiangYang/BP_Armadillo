//
// Created by yfx on 16-12-4.
//

#ifndef BP_BPNET_H
#define BP_BPNET_H
#include <iostream>
#include <armadillo>
#include "vector"
using namespace std;
using namespace arma;
//defination of input layer
typedef struct InputL
{
    mat Value,Samples;
}Input;
//defination of hidden layer
typedef struct HiddenL
{
    mat w,Dw,delta,Value,Response;//weight response
    double b,Db;//bias and its d
}Hidden;
//defination of output layer
typedef struct OutputL
{
    mat w,Dw,Value,Response,Labels,Label,delta;//weight matrix and others
}Out;
class BPnet
{
public:
    BPnet(mat Data,mat Label);//construct func
    BPnet(BPnet &net);//copy construct func
    void forward();//forward
    void backpropogation();//BP
    void train();
    void predict(mat Test);//predict func
    mat sigmoid(mat data);//sig func
    ~BPnet();//destructor
public:
    int OutNeuronNum,InputNeuronNum,HiddenLayerNum;
    double alpha,lambda,eplision,cost;//learning rate,weight decay,eplision and cost func
    vector<int>HiddenNeuronNum;
    Input *InputLayer;//in layer
    vector<Hidden*> HiddenLayer;//hidden layer
    Out *OutLayer;//outlayer
};


#endif //BP_BPNET_H