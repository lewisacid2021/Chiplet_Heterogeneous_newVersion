#include <cstring>
#include <fstream>
#include <iostream>

#include "apis_c.h"

constexpr int array_size=1024;

int idX, idY;

int main(int argc, char **argv) {

    idX = atoi(argv[1]);
    idY = atoi(argv[2]);

    int *arr = (int *)malloc(sizeof(int)*array_size);
    int *sum = (int *)malloc(sizeof(int));
    
    for(int i=0;i<array_size;i++)
    {
        arr[i] = i;
        InterChiplet::writeMemory(0,1,idX,idY,new InterChiplet::MemStruct((char *)(arr+i),i*sizeof(int),sizeof(int)));
    }

    memset(arr, 0, sizeof(int)*array_size);

    InterChiplet::readMemory(0,1,idX,idY,new InterChiplet::MemStruct((char *)arr,0x0,sizeof(int)*array_size));

    InterChiplet::sendMessage(1, 0, idX,idY, (void*)arr, sizeof(int)*array_size);
    InterChiplet::receiveMessage(idX,idY,1,0, (void*)sum, sizeof(int));

    std::cout << "[TEST] sum of array =" << *sum << std::endl;

    InterChiplet::stopMemory(0,1);
    return 0;
}
