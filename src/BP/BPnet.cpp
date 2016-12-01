#include "BPnet.h"

//���캯��
BPnet::BPnet(mat Data, mat Labels)
{
	cout <<"����������Ŀ��" << endl;
	cin >> HiddenLayerNum;
	//������Ԫ������ʼ��
	for (int i = 0; i < HiddenLayerNum; i++)
	{
		int temp;
		cout << "�����"<<1+i<<"��������Ԫ��Ŀ��" << endl;
		cin >> temp;
		HiddenNeuronNum.push_back(temp);//��vec
	}
	//��ʼ��������������
	cout << "�����������Ԫ��Ŀ��" << endl;
	cin >> OutNeuronNum;
	cout << "�����������Ԫ��Ŀ��" << endl;
	cin >> InputNeuronNum;
	err = 1; alpha = 0.9; lambda = 0.05; eplision = 0.001;//���ۺ�����ѧϰ�ʡ�Ȩ��˥��ϵ���뾫��
	//��ʼ����������
	InputLayer = new Input;  OutLayer = new Out;//�����ڴ�
	InputLayer->SampleData = Data;//������㴫��ѵ������,dataÿ��Ϊһ��ѵ������
	InputLayer->Value = Data.col(0);//ȡ��һ��ѵ��������ʼ��
	//�����ʼ��
	for (int i = 0; i < HiddenLayerNum; i++)
	{
		Hidden *tempHide = new Hidden;//ÿ��һ������ڵ��newһ���³�Ա,����ʼ����ɾ�pushback
		if (i == 0)//��һ��������Щ����
		{
			tempHide->bias = 0.f; tempHide->Db = 0.f;//����ƫִ������
			tempHide->w = randn(InputNeuronNum, HiddenNeuronNum.at(i));//��ʼ��һ��Ȩ�ؾ��󣬴�������������
			tempHide->Value = zeros(HiddenNeuronNum.at(i), 1);//�������һ������
			tempHide->delta = zeros(HiddenNeuronNum.at(i), 1);//�������һ������
			tempHide->Dw = zeros(InputNeuronNum, HiddenNeuronNum.at(i));//�������һ������
			tempHide->Response = zeros(HiddenNeuronNum.at(i), 1);//sig����
		}
		else
		{
			tempHide->bias = 0.f; tempHide->Db = 0.f;
			tempHide->w = randn(HiddenNeuronNum.at(i - 1), HiddenNeuronNum.at(i));//��ʼ��һ��Ȩ�ؾ��󣬴�������������
			tempHide->Value = zeros(HiddenNeuronNum.at(i), 1);//�������һ������
			tempHide->delta = zeros(HiddenNeuronNum.at(i), 1);//�������һ������
			tempHide->Dw = zeros(InputNeuronNum, HiddenNeuronNum.at(i));//�������һ������
			tempHide->Response = zeros(HiddenNeuronNum.at(i), 1);//sig����
		}
		HiddenLayer.push_back(tempHide);//����hiddenָ������
	}
	//������ʼ��
	OutLayer->Value = zeros(OutNeuronNum, 1);//�������һ������
	OutLayer->delta = zeros(OutNeuronNum, 1);//�в�����
	OutLayer->w = randn(HiddenNeuronNum.back(), OutNeuronNum);
	OutLayer->Response = zeros(OutNeuronNum, 1);//sig�����
	OutLayer->LabelData = Labels;
	OutLayer->Label = Labels.col(0);//�ѱ�ǩ��ֵ.��ǩÿһ����һ��������������ǩ��ÿһ�д����ǩ��һ��ά��
}
//�������캯��
BPnet::BPnet(BPnet &net)
{
	HiddenLayerNum = net.HiddenLayerNum;
	InputNeuronNum = net.InputNeuronNum;
	OutNeuronNum = net.OutNeuronNum;
	err = net.err; alpha = net.alpha; lambda = net.lambda; eplision = net.eplision;
	//���������Ԫ��Ŀ��ʼ��
	for (int i = 0; i < HiddenLayerNum; i++)
	{
		int tmp = net.HiddenNeuronNum.at(i);
		HiddenNeuronNum.push_back(tmp);
	}
	InputLayer = new Input;  OutLayer = new Out;//�����ڴ�
	InputLayer->SampleData = net.InputLayer->SampleData;//������㴫��ѵ������,dataÿ��Ϊһ��ѵ������
	InputLayer->Value = net.InputLayer->Value;//ȡ��һ��ѵ��������ʼ��
	//�����ʼ��
	for (int i = 0; i < HiddenLayerNum; i++)
	{
		Hidden *tempHide = new Hidden;//ÿ��һ������ڵ��newһ���³�Ա,����ʼ����ɾ�pushback
		tempHide->bias = net.HiddenLayer.at(i)->bias;
		tempHide->Db = net.HiddenLayer.at(i)->Db;
		tempHide->delta = net.HiddenLayer.at(i)->delta;
		tempHide->Dw = net.HiddenLayer.at(i)->Dw;
		tempHide->Response = net.HiddenLayer.at(i)->Response;
		tempHide->Value = net.HiddenLayer.at(i)->Value;
		tempHide->w = net.HiddenLayer.at(i)->w;
		HiddenLayer.push_back(tempHide);//����hiddenָ������
	}
	//������ʼ��
	OutLayer->Value = net.OutLayer->Value;//�������һ������
	OutLayer->delta = net.OutLayer->delta;//�в�����
	OutLayer->w = net.OutLayer->w;
	OutLayer->Response = net.OutLayer->Response;//sig�����
	OutLayer->LabelData = net.OutLayer->LabelData;
	OutLayer->Label = net.OutLayer->Label;//�ѱ�ǩ��ֵ.��ǩÿһ����һ��������������ǩ��ÿһ�д����ǩ��һ��ά��
}
//sigmoid ����
mat BPnet::sigmoid(mat vec)
{
	mat temp(vec.n_rows,1);
	for (int i = 0; i < vec.n_rows; i++)
	{
		temp(i, 0) = 1.0 / (1 + exp(-vec(i, 0)));
	}
	return temp;
}
//ǰ�򴫲�
void BPnet::forward()
{
	//���������ǰ�򴫲�
	for (int i = 0; i < HiddenLayerNum; i++)
	{
		if (i == 0)//��һ���������ݲ�һ��
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
	//�������ǰ�򴫲�
	OutLayer->Value = (HiddenLayer.at(HiddenLayerNum - 1)->Response.t()*OutLayer->w).t();
	OutLayer->Response = sigmoid(OutLayer->Value);
}
//���򴫲�
void BPnet::backproporgation()
{
	//����в���򴫲��������
	mat temp = OutLayer->Response - OutLayer->Label;
	err = as_scalar(temp.t()*temp); //������ۺ���(�ڻ�)
	OutLayer->delta =temp % (1 - OutLayer->Response) % OutLayer->Response;//������˻�������ʽ
	//����в���򴫲�����������
	for (int i = HiddenLayerNum - 1; i >= 0; i--)
	{
		if (i == HiddenLayerNum - 1)
		{
			mat sumdelta = OutLayer->w*OutLayer->delta;
			HiddenLayer.at(i)->delta =
				sumdelta%HiddenLayer.at(i)->Response%(1 - HiddenLayer.at(i)->Response);//������˻�������ʽ
		}
		else
		{
			mat sumdelta = HiddenLayer.at(i + 1)->w*HiddenLayer.at(i + 1)->delta;
			HiddenLayer.at(i)->delta =
				sumdelta%HiddenLayer.at(i)->Response%(1 - HiddenLayer.at(i)->Response);
		}
	}
	//�ݶȼ����ȴ����㿪ʼ
	for (int i = 0; i < HiddenLayerNum; i++)
	{
		if (i == 0)//��һ����������
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
	//�ٴ���������
	OutLayer->Dw = HiddenLayer.at(HiddenLayerNum - 1)->Response*OutLayer->delta.t();
}
//ѵ������
void BPnet::train()
{
	int epoch = 0, sampleNum = InputLayer->SampleData.n_cols;
	while (err>eplision)//���͵��ض�������²���ֹ
	{
		epoch++;
		cout << "epoch:  " << epoch << "  " << "cost:  " << err << "  " << endl;
		for (int i = 0; i < sampleNum; i++)
		{
			//�����뷴�򴫲�֮ǰ�������Out->label��Input->Value
			InputLayer->Value = InputLayer->SampleData.col(i);
			OutLayer->Label = OutLayer->LabelData.col(i);
			//����������ϣ�����ѵ��
			forward();
			backproporgation();//ǰ����򴫲����Ȩ�����ݶȾ���
			//��������
			for (int i = 0; i < HiddenLayerNum; i++)//ÿһ��epoch������������������ѵ��
			{
				HiddenLayer.at(i)->w -= alpha*HiddenLayer.at(i)->Dw / sampleNum;
				HiddenLayer.at(i)->bias -= alpha*HiddenLayer.at(i)->Db / sampleNum;
			}
			//���������
			OutLayer->w -= alpha*OutLayer->Dw / sampleNum;
		}
	}
}
//Ԥ�⺯��
void BPnet::predict(mat Data)
{
	mat sample(Data.n_rows, 1);
	//����Ԥ��
	for (int i = 0; i < Data.n_cols; i++)
	{
		//���һ��net
		BPnet newnet = *this;
		sample = (Data.col(i)).t();//�������
		for (int j = 0; j < HiddenLayerNum; j++)
		{
			//����ǰ�򴫲������Ӧ
			if (j == 0)//��һ���������ڲ�ͬ
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
		//���������
		newnet.OutLayer->Value = (newnet.HiddenLayer.at(HiddenLayerNum - 1)->Response.t()*newnet.OutLayer->w)
			+ newnet.HiddenLayer.at(HiddenLayerNum - 1)->bias;
			newnet.OutLayer->Response = sigmoid(newnet.OutLayer->Value);
		//������
		cout << "����" << i + 1 << "   " << "Ԥ����" << newnet.OutLayer->Response << endl;
	}
}
//��������
BPnet::~BPnet()
{

}