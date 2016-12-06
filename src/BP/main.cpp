
#include "BPnet.h"


int main()
{
    mat Sample(2, 4),Label(1,4);
    Sample << 1 << 1 << 0 << 0 << endr
           << 1 << 0 << 0 << 1 << endr;
    Label << 1 << 0 << 1 << 0 << endr;
    BPnet net(Sample,Label);
    //training
    net.train();
    //test
    mat Test(2, 4);
    Test << 0<< 0.99<< 0.1<< 1.1<<endr
         <<0.05<< 0.98<< 1.2<< 0.01<<endr;
    net.predict(Test);
    return 0;
}