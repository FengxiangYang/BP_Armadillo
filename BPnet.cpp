#include "BPnet.h"

//构造函数
BPnet::BPnet(mat Data, mat Labels)
{
	cout <<"输入隐层数目：" << endl;
	cin >> HiddenLayerNum;
	//隐层神经元参数初始化
	for (int i = 0; i < HiddenLayerNum; i++)
	{
		int temp;
		cout << "输入第"<<1+i<<"层隐层神经元数目：" << endl;
		cin >> temp;
		HiddenNeuronNum.push_back(temp);//入vec
	}
	//初始化其他基本参数
	cout << "输入输出层神经元数目：" << endl;
	cin >> OutNeuronNum;
	cout << "输入输入层神经元数目：" << endl;
	cin >> InputNeuronNum;
	err = 1; alpha = 0.9; lambda = 0.05; eplision = 0.001;//代价函数、学习率、权重衰减系数与精度
	//初始化输入层参数
	InputLayer = new Input;  OutLayer = new Out;//分配内存
	InputLayer->SampleData = Data;//给输入层传入训练参数,data每列为一个训练参数
	InputLayer->Value = Data.col(0);//取第一个训练参数初始化
	//隐层初始化
	for (int i = 0; i < HiddenLayerNum; i++)
	{
		Hidden *tempHide = new Hidden;//每有一个隐层节点就new一个新成员,待初始化完成就pushback
		if (i == 0)//第一层隐层有些特殊
		{
			tempHide->bias = 0.f; tempHide->Db = 0.f;//关于偏执的向量
			tempHide->w = randn(InputNeuronNum, HiddenNeuronNum.at(i));//初始化一个权重矩阵，待会儿付给输入层
			tempHide->Value = zeros(HiddenNeuronNum.at(i), 1);//给输入层一个向量
			tempHide->delta = zeros(HiddenNeuronNum.at(i), 1);//给输入层一个向量
			tempHide->Dw = zeros(InputNeuronNum, HiddenNeuronNum.at(i));//给输入层一个向量
			tempHide->Response = zeros(HiddenNeuronNum.at(i), 1);//sig激活
		}
		else
		{
			tempHide->bias = 0.f; tempHide->Db = 0.f;
			tempHide->w = randn(HiddenNeuronNum.at(i - 1), HiddenNeuronNum.at(i));//初始化一个权重矩阵，待会儿付给输入层
			tempHide->Value = zeros(HiddenNeuronNum.at(i), 1);//给输入层一个向量
			tempHide->delta = zeros(HiddenNeuronNum.at(i), 1);//给输入层一个向量
			tempHide->Dw = zeros(InputNeuronNum, HiddenNeuronNum.at(i));//给输入层一个向量
			tempHide->Response = zeros(HiddenNeuronNum.at(i), 1);//sig激活
		}
		HiddenLayer.push_back(tempHide);//进入hidden指针数组
	}
	//输出层初始化
	OutLayer->Value = zeros(OutNeuronNum, 1);//给输出层一个向量
	OutLayer->delta = zeros(OutNeuronNum, 1);//残差向量
	OutLayer->w = randn(HiddenNeuronNum.back(), OutNeuronNum);
	OutLayer->Response = zeros(OutNeuronNum, 1);//sig激活函数
	OutLayer->LabelData = Labels;
	OutLayer->Label = Labels.col(0);//把标签赋值.标签每一列是一个单独的样本标签，每一行代表标签的一个维度
}
//拷贝构造函数
BPnet::BPnet(BPnet &net)
{
	HiddenLayerNum = net.HiddenLayerNum;
	InputNeuronNum = net.InputNeuronNum;
	OutNeuronNum = net.OutNeuronNum;
	err = net.err; alpha = net.alpha; lambda = net.lambda; eplision = net.eplision;
	//完成隐层神经元数目初始化
	for (int i = 0; i < HiddenLayerNum; i++)
	{
		int tmp = net.HiddenNeuronNum.at(i);
		HiddenNeuronNum.push_back(tmp);
	}
	InputLayer = new Input;  OutLayer = new Out;//分配内存
	InputLayer->SampleData = net.InputLayer->SampleData;//给输入层传入训练参数,data每列为一个训练参数
	InputLayer->Value = net.InputLayer->Value;//取第一个训练参数初始化
	//隐层初始化
	for (int i = 0; i < HiddenLayerNum; i++)
	{
		Hidden *tempHide = new Hidden;//每有一个隐层节点就new一个新成员,待初始化完成就pushback
		tempHide->bias = net.HiddenLayer.at(i)->bias;
		tempHide->Db = net.HiddenLayer.at(i)->Db;
		tempHide->delta = net.HiddenLayer.at(i)->delta;
		tempHide->Dw = net.HiddenLayer.at(i)->Dw;
		tempHide->Response = net.HiddenLayer.at(i)->Response;
		tempHide->Value = net.HiddenLayer.at(i)->Value;
		tempHide->w = net.HiddenLayer.at(i)->w;
		HiddenLayer.push_back(tempHide);//进入hidden指针数组
	}
	//输出层初始化
	OutLayer->Value = net.OutLayer->Value;//给输出层一个向量
	OutLayer->delta = net.OutLayer->delta;//残差向量
	OutLayer->w = net.OutLayer->w;
	OutLayer->Response = net.OutLayer->Response;//sig激活函数
	OutLayer->LabelData = net.OutLayer->LabelData;
	OutLayer->Label = net.OutLayer->Label;//把标签赋值.标签每一列是一个单独的样本标签，每一行代表标签的一个维度
}
//sigmoid 函数
mat BPnet::sigmoid(mat vec)
{
	mat temp(vec.n_rows,1);
	for (int i = 0; i < vec.n_rows; i++)
	{
		temp(i, 0) = 1.0 / (1 + exp(-vec(i, 0)));
	}
	return temp;
}
//前向传播
void BPnet::forward()
{
	//误差在隐层前向传播
	for (int i = 0; i < HiddenLayerNum; i++)
	{
		if (i == 0)//第一层输入数据不一样
		{
			HiddenLayer.at(i)->Value = (InputLayer->Value.t()*HiddenLayer.at(i)->w).t()+
				HiddenLayer.at(i)->bias;
			HiddenLayer.at(i)->Response = sigmoid(HiddenLayer.at(i)->Value);
		}
		else
		{
			HiddenLayer.at(i)->Value = (HiddenLayer.at(i - 1)->Response.t()*HiddenLayer.at(i)->w).t() +
				HiddenLayer.at(i)->bias;
			HiddenLayer.at(i)->Response = sigmoid(HiddenLayer.at(i)->Value);
		}
	}
	//在输出层前向传播
	OutLayer->Value = (HiddenLayer.at(HiddenLayerNum - 1)->Response.t()*OutLayer->w).t();
	OutLayer->Response = sigmoid(OutLayer->Value);
}
//后向传播
void BPnet::backproporgation()
{
	//计算残差。反向传播到输出层
	mat temp = OutLayer->Response - OutLayer->Label;
	err = as_scalar(temp.t()*temp); //计算代价函数(内积)
	OutLayer->delta =temp % (1 - OutLayer->Response) % OutLayer->Response;//阿达马乘积调用形式
	//计算残差。反向传播到各个隐层
	for (int i = HiddenLayerNum - 1; i >= 0; i--)
	{
		if (i == HiddenLayerNum - 1)
		{
			mat sumdelta = OutLayer->w*OutLayer->delta;
			HiddenLayer.at(i)->delta =
				sumdelta%HiddenLayer.at(i)->Response%(1 - HiddenLayer.at(i)->Response);//阿达马乘积调用形式
		}
		else
		{
			mat sumdelta = HiddenLayer.at(i + 1)->w*HiddenLayer.at(i + 1)->delta;
			HiddenLayer.at(i)->delta =
				sumdelta%HiddenLayer.at(i)->Response%(1 - HiddenLayer.at(i)->Response);
		}
	}
	//梯度计算先从隐层开始
	for (int i = 0; i < HiddenLayerNum; i++)
	{
		if (i == 0)//第一层总是特殊
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
	//再从输出层计算
	OutLayer->Dw = HiddenLayer.at(HiddenLayerNum - 1)->Response*OutLayer->delta.t();
}
//训练函数
void BPnet::train()
{
	int epoch = 0, sampleNum = InputLayer->SampleData.n_cols;
	while (err>eplision)//降低到特定误差以下才终止
	{
		epoch++;
		cout << "epoch:  " << epoch << "  " << "cost:  " << err << "  " << endl;
		for (int i = 0; i < sampleNum; i++)
		{
			//正向与反向传播之前必须更新Out->label和Input->Value
			InputLayer->Value = InputLayer->SampleData.col(i);
			OutLayer->Label = OutLayer->LabelData.col(i);
			//更新数据完毕，进行训练
			forward();
			backproporgation();//前向后向传播获得权重与梯度矩阵
			//更新隐层
			for (int i = 0; i < HiddenLayerNum; i++)//每一个epoch都将所有样本进行了训练
			{
				HiddenLayer.at(i)->w -= alpha*HiddenLayer.at(i)->Dw / sampleNum;
				HiddenLayer.at(i)->bias -= alpha*HiddenLayer.at(i)->Db / sampleNum;
			}
			//更新输出层
			OutLayer->w -= alpha*OutLayer->Dw / sampleNum;
		}
	}
}
//预测函数
void BPnet::predict(mat Data)
{
	mat sample(Data.n_rows, 1);
	//进行预测
	for (int i = 0; i < Data.n_cols; i++)
	{
		//深拷贝一个net
		BPnet newnet = *this;
		sample = (Data.col(i)).t();//获得特征
		for (int j = 0; j < HiddenLayerNum; j++)
		{
			//进行前向传播求得响应
			if (j == 0)//第一层总是与众不同
			{
				newnet.HiddenLayer.at(j)->Value = (sample*newnet.HiddenLayer.at(j)->w).t()
					+ newnet.HiddenLayer.at(j)->bias;
				newnet.HiddenLayer.at(j)->Response = sigmoid(newnet.HiddenLayer.at(j)->Value);
			}
			else
			{
				newnet.HiddenLayer.at(j)->Value = (newnet.HiddenLayer.at(j - 1)->Response.t()*newnet.HiddenLayer.at(j)->w).t()
					+ newnet.HiddenLayer.at(j)->bias;
				newnet.HiddenLayer.at(j)->Response = sigmoid(newnet.HiddenLayer.at(j)->Value);
			}
		}
		//误差传到输出层
		newnet.OutLayer->Value = (newnet.HiddenLayer.at(HiddenLayerNum - 1)->Response.t()*newnet.OutLayer->w)
			+ newnet.HiddenLayer.at(HiddenLayerNum - 1)->bias;
			newnet.OutLayer->Response = sigmoid(newnet.OutLayer->Value);
		//输出结果
		cout << "样本" << i + 1 << "   " << "预测结果" << newnet.OutLayer->Response << endl;
	}
}
//析构函数
BPnet::~BPnet()
{

}