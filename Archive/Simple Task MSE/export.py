import rclone
import os
import yaml
import shutil

### SWTICH WHEN FUNCTIONAL

rclone_conf = os.environ['RCLONE_CONF']
with open(rclone_conf) as f:
    cfg = f.read()
    print(cfg)

### PARMATERS YAML
OUT_DIR = os.path.join(os.environ['DM_T'])
path = os.path.join(OUT_DIR,'vassilis_out')

# with open(os.path.join(path,'model_slurm_config.yml'), 'r') as cfg_file:
#     slurm_dict = yaml.safe_load(cfg_file)
# num_sample = slurm_dict['num_sample']    

path_local = os.path.join(os.environ['DM_T'],'vassilis_out')
path_remote = 'sgd:BIL_Figures'
# path_remote = os.path.join('sgd:BIL_Figures') #'model_out_{}'.format(num_sample))
rclone.with_config(cfg).copy(path_local, path_remote, flags=['-Pv','--transfers=10']) #,'--exclude=main/**','--exclude=supp/**'])


### FOR NOW JUST DELETING FOLDER , REMOVE ONCE RCLONE WORKING
shutil.rmtree(os.path.join(os.environ['DM_T'],'vassilis_out')) # shutil.rmtree(os.environ['DM_T'],'vassilis_out')
