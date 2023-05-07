#!/bin/bash
# Download train split of dataset ThePile.
# Download dataset shards, decompress them and remove the compressed files.

local_dir=/mnt/reginald/thepile/train
for i in {00..29}
do
    filename=$i.jsonl.zst
    curl https://the-eye.eu/public/AI/pile/train/$filename -o $local_dir/$filename
    zstd -d $local_dir/$filename
    rm $local_dir/$filename
done
