# Solver for the Ax=b problem

This concerne mainly the problem where the number of rows of A is smaller than the number of columns, and x is sparse.

<<<<<<< HEAD
sudo apt update
sudo apt upgrade
sudo apt install cmake libopenblas-dev liblapack-dev

wget http://sourceforge.net/projects/arma/files/armadillo-9.800.4.tar.xz

cd arma*
cmake .
make
sudo make install
=======
```
rm -rf Debug ; mkdir Debug ; cd Debug ; cmake -GNinja -DCMAKE_BUILD_TYPE=Debug .. ; ninja
```

```
rm -rf Release ; mkdir Release ; cd Release ; cmake -GNinja -DCMAKE_BUILD_TYPE=Release .. ; ninja
```

```
rm -rf Debug ; mkdir Debug ; cd Debug ; cmake -DCMAKE_BUILD_TYPE=Debug .. ; make
```

```
rm -rf Release ; mkdir Release ; cd Release ; cmake -DCMAKE_BUILD_TYPE=Release .. ; make
```
>>>>>>> a32dd76250b9f6caa10fd07f944505e42448baeb
