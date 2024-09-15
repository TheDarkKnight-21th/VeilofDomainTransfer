# pruning book download

cd pytorch-image-models/pruning_book/
mkdir feature_book
cd feature_book

gdown 1fNwJZ2KbX1UMCPtddpAFzFvGwv4bU9sZ
gdown 19qio_rgS62gzrfNIlMdM2FzXgvhxXqrM
gdown 15uaN1ZQkpjx1RxbUCjVBqkP5N19H70m9
gdown 11Th1-gCGgJq1d_b8HcCojvdKKKRN_Ge6

cd ../../../


# bash ./download.sh 실행 후 data directory

# dataset/
# ├─ imagenet21k/
# │  └─ train
# ├─ imagenet21k_resized/
# │  ├─ imagenet21k_small_classes 
# │  ├─ imagenet21k_train  
# │  └─ imagenet21k_val
# └─ da
#     ├─ domainnet
#     ├─ office-home
#     └─ cub


mkdir dataset
cd dataset 

# install library for google drive download
pip install gdown

# Downstream dataset downlaod

mkdir da
cd da

#Office-home
mkdir office-home
cd office-home

#download dataset
gdown 1yNwbJcDvGpYS4IifywUWT-morLem5Bcj
#download imagelist
gdown 1dYuxRCLyxT9_JWIrff251WHj1zurcxQh
pwd
#extract the dataset
unzip '*.zip'

mv 'Real World' Real_World

cd ..
pwd

#domainnet
mkdir domainnet
cd domainnet

#download the image-list
gdown 1ADy--fjpqG4vggQ_IZcI1dl_TW_VxcFI

#download the dataset
wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip
wget http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip

# extract the dataset
for z in *.zip; do unzip "$z"; done

cd ..
pwd

#CUB

#download the dataset
gdown --folder --id 1t24AEktRRPdsjqEXq5hjZAqGV2WpSBpH?

cd cub

#extract the data
tar -xvf *.tgz
unzip '*.zip'

cd CUB_200_2011
rm -rf attributes
rm -rf parts

mv ./images/* ./
rm -rf images

cd ../../../

pwd

#download inagenet21k-resized
# wget https://www.image-net.org/data/imagenet21k_resized.tar.gz

# tar -xvf imagenet21k_resized.tar.gz

#download inagenet21k
mkdir imagenet21k
cd imagenet21k

wget https://www.image-net.org/data/winter21_whole.tar.gz

tar -xvf winter21_whole.tar.gz

mkdir train

target_dir=./train/

for tar in ./winter21_whole/*.tar; do
  filename=$(basename $tar .tar)  
  mkdir -p $target_dir/$filename
  tar -xf $tar -C $target_dir/$filename
done


rm -rf winter21_whole





