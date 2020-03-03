#!/bin/bash

if ! [ -f $1 ]; then
    echo "Error. File $1 doesn't exist!"
    exit 1
fi

for i in {1..3};
do
    $1 1000000000 >> out.txt
    echo "\n" >> out.txt
done

