#!/bin/bash
mkdir '/content/yeastract'
mv '/content/Regulatory-Gene-Network-Inferance-from-gene-exepression-data/yeastract_w_GT.tar.bz2' '/content/yeastract/yeastract_w_GT.tar.bz2'
rm -R '/content/Regulatory-Gene-Network-Inferance-from-gene-exepression-data/'
cd '/content/yeastract'
tar -xf '/content/yeastract/yeastract_w_GT.tar.bz2'
cd ..