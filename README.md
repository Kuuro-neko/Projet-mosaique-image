# Mosaique images

## Build and run the project

```bash
./build.sh && ./launch.sh <image_in> <image_dataset> <bloc_size> [method] [stats_param]
```
Example : 
```bash 
./build.sh && ./launch.sh img/chat_320.jpg dataset/train 8 0 11000
```
method (optional) : 
 - 0 for statistics (default if not given)
 - 1 for alignment

stats_param (used for statistics method, optional) : \
Bit array of length 5 : abcde, default if not given is 11000
 - a : 1 to use mean color, 0 to not use it
 - b : 1 to use color variance, 0 to not use it
 - c : 1 to use skewness, 0 to not use it
 - d : 1 to use energy, 0 to not use it
 - e : 1 to use the long but best unique blocs algorithm, 0 for the naive one
 