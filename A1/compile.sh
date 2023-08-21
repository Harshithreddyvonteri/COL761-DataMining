module load compiler/gcc/7.1.0/compilervars
module load compiler/gcc/11.2.0
g++ fptree.cpp -o fptree -O3
g++ decomp.cpp -o decomp -O3
