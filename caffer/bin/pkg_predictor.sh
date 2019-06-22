#!/bin/bash
date=`date +%Y-%m-%dxx%H.%M.%S`

rm -rf pkg_predictor
mkdir pkg_predictor

cd pkg_predictor
svn export svn://localhost/codes/libs/py/lib
svn export svn://localhost/codes/ai/caffer
rm caffer/rec* caffer/classifer caffer/bin caffer/test caffer/pytrain_test -rf
rm -rf caffer/netbuilder.py caffer/*.prototxt
#cp /.jingyi/codes/caffer/classifer2/predictor/models caffer/classifer2/predictor/ -r
cd ..

tar_name="pkg_predictor_$date.tgz"
tar -czvf $tar_name pkg_predictor
rm -rf pkg_predictor

# $Id: $


