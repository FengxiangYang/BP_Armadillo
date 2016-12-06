//
// Created by yfx on 16-12-4.
//

#include "BPnet.h"

//construc func
BPnet::BPnet(mat Data, mat Label)
{
    cout<<"How many Hidden layers in your network?"<<endl;
    cin>>HiddenLayerNum;
    InputLayer=new Input;OutLayer=new Out;
    //init input layer
    InputLayer->Samples=Data;
    InputLayer->Value=Data.col(0);//init by the first sample
    InputNeuronNum=Data.n_rows;
    //init hidden layer
    for(int i=0;i<HiddenLayerNum;i++)
    {
        int temp;
        Hidden *Hidetemp=new Hidden;
        cout<<"Total neuron numbers in hidden layer"<<i+1<<":"<<endl;
        cin>>temp;
        HiddenNeuronNum.push_back(temp);
        if(i==0)
        {
            Hidetemp->Value=zeros(temp,1);
            Hidetemp->Response=Hidetemp->Value;
            Hidetemp->b=0;Hidetemp->Db=0;
            Hidetemp->delta=zeros(temp,1);
            Hidetemp->w=randn(InputNeuronNum,temp);
            Hidetemp->Dw=Hidetemp->w;
        }
        else
        {
            Hidetemp->Value=zeros(temp,1);
            Hidetemp->Response=Hidetemp->Value;
            Hidetemp->b=0;Hidetemp->Db=0;
            Hidetemp->delta=zeros(temp,1);
            Hidetemp->w=randn(HiddenNeuronNum.at(i-1),temp);
            Hidetemp->Dw=Hidetemp->w;
        }
        HiddenLayer.push_back(Hidetemp);
    }
    //init input layer neuron num and out ~~~
    InputNeuronNum=Data.n_rows;OutNeuronNum=Label.n_rows;
    //init output
    OutLayer->w=randn(HiddenNeuronNum.at(HiddenLayerNum-1),OutNeuronNum);
    OutLayer->Dw=OutLayer->w;
    OutLayer->Value=zeros(OutNeuronNum,1);
    OutLayer->Response=OutLayer->Value;
    OutLayer->Labels=Label;
    OutLayer->Label=Label.col(0);
    OutLayer->delta=zeros(OutNeuronNum,1);
    //init others
    alpha=0.9;lambda=0.0;eplision=0.0001;cost=1.0;
}
//copy constructor
BPnet::BPnet(BPnet &net)
{
    //copy layer params
    OutLayer=net.OutLayer;
    OutNeuronNum=net.OutNeuronNum;
    HiddenLayerNum=net.HiddenLayerNum;
    //basic params
    alpha=net.alpha;
    cost=net.cost;
    lambda=net.lambda;
    eplision=net.eplision;
    //copy inputlayers
    InputLayer=new Input;OutLayer=new Out;
    for (int i = 0; i < HiddenLayerNum; i++)
    {
        int tmp = net.HiddenNeuronNum.at(i);
        HiddenNeuronNum.push_back(tmp);
    }
    InputLayer->Samples = net.InputLayer->Samples;
    InputLayer->Value = net.InputLayer->Value;
    //copy hidden layers
    for(int i=0;i<HiddenLayerNum;i++)
    {
        Hidden *tempHide = new Hidden;//malloc
        tempHide->b = net.HiddenLayer.at(i)->b;
        tempHide->Db = net.HiddenLayer.at(i)->Db;
        tempHide->delta = net.HiddenLayer.at(i)->delta;
        tempHide->Dw = net.HiddenLayer.at(i)->Dw;
        tempHide->Response = net.HiddenLayer.at(i)->Response;
        tempHide->Value = net.HiddenLayer.at(i)->Value;
        tempHide->w = net.HiddenLayer.at(i)->w;
        HiddenLayer.push_back(tempHide);//push
    }
    //copy outputlayers
    OutLayer->Value = net.OutLayer->Value;
    OutLayer->delta = net.OutLayer->delta;
    OutLayer->w = net.OutLayer->w;
    OutLayer->Response = net.OutLayer->Response;
    OutLayer->Labels = net.OutLayer->Labels;
    OutLayer->Label = net.OutLayer->Label;
}
//sigmoid func
mat BPnet::sigmoid(mat vec)
{
    mat temp(vec.n_rows,1);
    for (int i = 0; i < vec.n_rows; i++)
    {
        temp(i, 0) = 1.0 / (1 + exp(-vec(i, 0)));
    }
    return temp;
}
//forward
void BPnet::forward()
{
    for(int i=0;i<HiddenLayerNum;i++)
    {
        if(i==0)
        {
            HiddenLayer.at(i)->Value=(InputLayer->Value.t()*HiddenLayer.at(i)->w).t()+HiddenLayer.at(i)->b;
            HiddenLayer.at(i)->Response=sigmoid(HiddenLayer.at(i)->Value);
        }
        else
        {
            HiddenLayer.at(i)->Value=(HiddenLayer.at(i-1)->Value.t()*HiddenLayer.at(i)->w).t()+HiddenLayer.at(i)->b;
            HiddenLayer.at(i)->Response=sigmoid(HiddenLayer.at(i)->Value);
        }
    }
    //spread in the output layer
    OutLayer->Value = (HiddenLayer.at(HiddenLayerNum - 1)->Response.t()*OutLayer->w).t();
    OutLayer->Response = sigmoid(OutLayer->Value);
}
//BP
void BPnet::backpropogation()
{
    //compute delta
    mat temp = OutLayer->Response - OutLayer->Label;
    cost = as_scalar(temp.t()*temp); //cost
    OutLayer->delta =temp % (1 - OutLayer->Response) % OutLayer->Response;//armadrillo way of .*
    //compute delta and spread to hidden layer
    for (int i = HiddenLayerNum - 1; i >= 0; i--)
    {
        if (i == HiddenLayerNum - 1)
        {
            mat sumdelta = OutLayer->w*OutLayer->delta;
            HiddenLayer.at(i)->delta =
                    sumdelta%HiddenLayer.at(i)->Response%(1 - HiddenLayer.at(i)->Response);//armadrillo way of .*
        }
        else
        {
            mat sumdelta = HiddenLayer.at(i + 1)->w*HiddenLayer.at(i + 1)->delta;
            HiddenLayer.at(i)->delta =
                    sumdelta%HiddenLayer.at(i)->Response%(1 - HiddenLayer.at(i)->Response);
        }
    }
    //compute deverate
    for (int i = 0; i < HiddenLayerNum; i++)
    {
        if (i == 0)//special for the first layer
        {
            HiddenLayer.at(i)->Dw = InputLayer->Value *HiddenLayer.at(i)->delta.t();
            HiddenLayer.at(i)->Db = as_scalar(sum(HiddenLayer.at(i)->delta));
        }
        else
        {
            HiddenLayer.at(i)->Dw = HiddenLayer.at(i - 1)->Response*HiddenLayer.at(i)->delta.t();
            HiddenLayer.at(i)->Db = as_scalar(sum(HiddenLayer.at(i)->delta));
        }
    }
    //compute output layer
    OutLayer->Dw = HiddenLayer.at(HiddenLayerNum - 1)->Response*OutLayer->delta.t();
}
//training
void BPnet::train()
{
    int epoch = 0, sampleNum = InputLayer->Samples.n_cols;
    while (cost>eplision)//loop until<ep
    {
        epoch++;
        cout << "epoch:  " << epoch << "  " << "cost:  " << cost << "  " << endl;
        for (int i = 0; i < sampleNum; i++)
        {
            //updae label and Input->Value
            InputLayer->Value = InputLayer->Samples.col(i);
            OutLayer->Label = OutLayer->Labels.col(i);
            //start training
            forward();
            backpropogation();//get weight mat
            //update hidden layer
            for (int i = 0; i < HiddenLayerNum; i++)//loop
            {
                HiddenLayer.at(i)->w -= alpha*HiddenLayer.at(i)->Dw / sampleNum;
                HiddenLayer.at(i)->b -= alpha*HiddenLayer.at(i)->Db / sampleNum;
            }
            //update out
            OutLayer->w -= alpha*OutLayer->Dw / sampleNum;
        }
    }
}
//predict
void BPnet::predict(mat Test)
{
    mat sample(Test.n_rows, 1);
    //starting predict
    for (int i = 0; i < Test.n_cols; i++)
    {
        //deep copy
        BPnet newnet = *this;
        sample = (Test.col(i)).t();//get feature
        for (int j = 0; j < HiddenLayerNum; j++)
        {
            //forward to gain response
            if (j == 0)//different at the first layer
            {
                newnet.HiddenLayer.at(j)->Value = (sample*newnet.HiddenLayer.at(j)->w).t()
                                                  + newnet.HiddenLayer.at(j)->b;
                newnet.HiddenLayer.at(j)->Response = sigmoid(newnet.HiddenLayer.at(j)->Value);
            }
            else
            {
                newnet.HiddenLayer.at(j)->Value = (newnet.HiddenLayer.at(j - 1)->Response.t()*newnet.HiddenLayer.at(j)->w).t()
                                                  + newnet.HiddenLayer.at(j)->b;
                newnet.HiddenLayer.at(j)->Response = sigmoid(newnet.HiddenLayer.at(j)->Value);
            }
        }
        //spread to out layer
        newnet.OutLayer->Value = (newnet.HiddenLayer.at(HiddenLayerNum - 1)->Response.t()*newnet.OutLayer->w)
                                 + newnet.HiddenLayer.at(HiddenLayerNum - 1)->b;
        newnet.OutLayer->Response = sigmoid(newnet.OutLayer->Value);
        //output the results
        cout << "sample" << i + 1 << "   " << "result" << newnet.OutLayer->Response << endl;
    }
}
//destructor
BPnet::~BPnet()
{

}