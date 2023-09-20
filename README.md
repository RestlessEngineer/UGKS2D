# UGKS
A two-dimensional unified gas-kinetic scheme (UGKS) for calculations on structured quadrilateral computational meshes.
### parallelism
This code utilizes OPENMP by default

### how to build
```
makedir build
cd build
cmake ..
make 
```
### examples

go to Release or Debug folder. You may launch every example with a simple command. Just put "example_file" path with the help of --init option. For box example it will be look like:
```
./UGKS2D -i "box/init.json"
```
All results will be saved in .plt format. You may open them with the help ParaView.


### options

You may setting up output, console output information and number of threads. For more information just launch --help option:
```
./UGKS2D --help
```

### using libraries
- **Eigen** for matrix calculations
