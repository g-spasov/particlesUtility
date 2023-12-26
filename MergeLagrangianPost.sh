#!bin/bash

E=$1

cd $E

# for i in $(ls)
# do
#     cd $i
rm collectedData.txt
for j in $(ls)
do
    cat $j/*.post >> collectedData.txt
done
#     cd ..
# done