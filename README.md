# Solver for the Ax=b problem

This concerne mainly the problem where the number of rows of A is smaller than the number of columns, and x is sparse.

sudo apt update
sudo apt upgrade
sudo apt install cmake libopenblas-dev liblapack-dev

wget http://sourceforge.net/projects/arma/files/armadillo-9.800.4.tar.xz

cd arma*
cmake .
make
sudo make install