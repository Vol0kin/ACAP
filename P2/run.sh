#!/bin/bash

if ! [ -f $1 ]; then
    echo "Error. File $1 doesn't exist!"
    exit 1
fi

echo "-------------------Resultados para $2 procesos-----------------------" >> out.txt

for i in {1..3};
do
    mpirun -hostfile myhostfile -np $2 $1 1000000000 >> out.txt
    echo "\n" >> out.txt
done

