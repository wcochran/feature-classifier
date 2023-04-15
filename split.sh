#!/bin/bash

CSV="s17-features.csv"
OUT="s17-features-"
#N=250000
N=485000

gsplit -d -l $N $CSV $OUT

H="N,IMGNAME,IMGID,I,KX,KY,A11,A12,A21,A22,MATCHES,INLIERS,HASPT3D,DESC"

for F in ${OUT}??; do
    echo $H | cat - $F > $F.csv
done

rm ${OUT}??

echo "Note : ${OUT}00.csv has an extra header line"
