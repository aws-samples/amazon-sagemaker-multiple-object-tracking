#!/bin/bash

set -e

if [ "$#" -ne 3 ]; then
    echo "usage: $0 <s3-bucket-name>"
    exit 1
fi

S3_BUCKET=$1
DATA_VERSION=$2
HALF_VAL=$3
S3_PREFIX="fairmot/sagemaker/input"

# Stage directory must be on EBS volume with 100 GB available space
STAGE_DIR=$(pwd)/datasets

echo "Create stage directory: $STAGE_DIR"

if [ ! -d "$STAGE_DIR" ]; then
    mkdir -p $STAGE_DIR
fi

zip_file=$STAGE_DIR/$DATA_VERSION.zip

if test -f "$zip_file"; then
    echo "$zip_file exists."
else
    wget -O $zip_file https://motchallenge.net/data/$DATA_VERSION.zip
fi

unzip_dir=$STAGE_DIR/$DATA_VERSION
if [ -d "$unzip_dir" ]; then
    echo "$unzip_dir exists."
    rm -r $unzip_dir
fi

echo "Extracting $zip_file"
unzip -o $zip_file  -d $STAGE_DIR | awk 'BEGIN {ORS="="} {if(NR%1000==0)print "="}'
echo "Done."

# Unzip $DATA_VERSION.zip
result_dir=$STAGE_DIR/train
if [ -d "$result_dir" ]; then
    echo "$result_dir exists."
    rm -r $result_dir
fi

images_dir=$result_dir/$DATA_VERSION/images
mkdir -p $images_dir
mv $STAGE_DIR/$DATA_VERSION/train $images_dir

rm -r $STAGE_DIR/$DATA_VERSION/test
rm $zip_file

echo "Generate labels for $DATA_VERSION"
python gen_labels.py --dataset-dir $STAGE_DIR/train --dataset-name $DATA_VERSION

echo "Generate path for $DATA_VERSION"
python gen_path.py --dataset-dir $STAGE_DIR/train --dataset-name $DATA_VERSION --half-val $HALF_VAL

# Prepare pretrained model
pip install gdown
pretrained_dir=$STAGE_DIR/train/pretrained-models
if [ -d "$pretrained_dir" ]; then
    echo "$pretrained_dir exists."
    rm -r $pretrained_dir
fi
mkdir -p $pretrained_dir
gdown https://drive.google.com/u/0/uc?id=1iqRQjsG9BawIl8SlFomMg5iwkb6nqSpi -O $pretrained_dir/fairmot_dla34.pth

echo "`date`: Uploading extracted files to s3://$S3_BUCKET/$S3_PREFIX/train [ eta 12 minutes ]"
aws s3 cp --recursive $STAGE_DIR/train s3://$S3_BUCKET/$S3_PREFIX/train | awk 'BEGIN {ORS="="} {if(NR%100==0)print "="}'

echo "Done."

echo "Delete stage directory: $STAGE_DIR"
rm -rf $STAGE_DIR
echo "Success."
