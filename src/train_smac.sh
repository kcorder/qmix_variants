#!/bin/bash

NS=1
MAP="2c_vs_64zg"

python main.py --config=qmix --env-config=sc2 with env_args.map_name=3m & 
sleep 1
python main.py --config=qmix --env-config=sc2 with env_args.map_name=3m &
sleep 1 

python main.py --config=qmix --env-config=sc2 with env_args.map_name=$MAP & 
sleep 1
python main.py --config=qmix --env-config=sc2 with env_args.map_name=$MAP &
sleep 1 

#python main.py --config=qmix_nops --env-config=sc2 with env_args.map_name=$MAP & 
#sleep 1 
#python main.py --config=qmix_nops --env-config=sc2 with env_args.map_name=$MAP &
#sleep 1 

python main.py --config=qmix_fc_nops --env-config=sc2 with env_args.map_name=$MAP & 
sleep 1 
python main.py --config=qmix_fc_nops --env-config=sc2 with env_args.map_name=$MAP &
sleep 1 

python main.py --config=qmix_fc_no_ps --env-config=sc2 with env_args.map_name=3m & 
sleep 1
python main.py --config=qmix_fc_no_ps --env-config=sc2 with env_args.map_name=3m & 
sleep 1

#python main.py --config=qmix_fc --env-config=sc2 with env_args.map_name=$MAP &
#sleep 1 
#python main.py --config=qmix_fc --env-config=sc2 with env_args.map_name=$MAP &
#sleep 1 

wait 
