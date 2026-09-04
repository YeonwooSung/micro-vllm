rm -rf report1.nsys-rep
rm -rf report1.sqlite
./test.sh
nsys profile ./build/micro-vllm-cuda --output="report1.nsys-rep"
nsys stats report1.nsys-rep