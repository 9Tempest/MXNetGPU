#!/bin/bash

if [ "$#" -ne 0 ]; then
        echo "Usage: /scratch/eecs471f24_class_root/eecs471f24_class/$USER/final_project/submit_code.sh"
        exit 1
fi

chmod 700 new-forward.cuh
cp -f new-forward.cuh /scratch/eecs471f24_class_root/eecs471f24_class/all_sub/$USER/final_project/$USER.cuh
setfacl -m u:"joeberns":rwx /scratch/eecs471f24_class_root/eecs471f24_class/all_sub/$USER/final_project/$USER.cuh
setfacl -m u:"aryanj":rwx /scratch/eecs471f24_class_root/eecs471f24_class/all_sub/$USER/final_project/$USER.cuh
setfacl -m u:"reetudas":rwx /scratch/eecs471f24_class_root/eecs471f24_class/all_sub/$USER/final_project/$USER.cuh
