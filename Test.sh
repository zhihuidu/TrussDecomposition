echo "./Test.sh inputfile loopnum"
rm -f tmpm
for (( c=0; c<$2;c++))
do
echo "./myminsearch < $1"
./minsearch < $1 2>>tmpm
done
./ExeTime.sh tmpm >tmp

sum=$(paste -sd+ tmp | bc)
echo "$sum / $2" | bc

rm -f tmpc
for (( c=0; c<$2;c++))
do
echo "./truss_decomposition_parallel < $1"
./minsearch < $1 2>>tmpc
done
./ExeTime.sh tmpc >tmp

sum=$(paste -sd+ tmp | bc)
echo "$sum / $2" | bc


rm -f tmpm
for (( c=0; c<$2;c++))
do
echo "./myminsearch < $1"
./minsearch < $1 2>>tmpm
done
./ExeTime.sh tmpm >tmp

sum=$(paste -sd+ tmp | bc)
echo "$sum / $2" | bc
