#! /bin/bash

IMG=multif_v25.img

singularity create --size 10240 $IMG
singularity bootstrap $IMG multif.def

# Script to shrink image from 
# https://github.com/singularityware/singularity/issues/623

image=$IMG
stripped_img=`tempfile --directory=.`
tail -n +2 $image > $stripped_img
e2fsck -f $stripped_img
resize2fs -M $stripped_img
shrunk_img=`tempfile --directory=.`
head -n 1 $image > $shrunk_img
cat $stripped_img >> $shrunk_img
rm $stripped_img
mv $shrunk_img $image
chmod a+x $image

