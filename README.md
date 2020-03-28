# Solver for the Ax=b problem

This concerne mainly the problem where the number of rows of A is smaller than the number of columns, and x is sparse.

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