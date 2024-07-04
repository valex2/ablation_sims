import yaml
import os

num_child = int(os.environ['NUM_CHILD'])
RUNNER = os.environ['RUNNER']

config_runner = {'default':
                  {'image': 'us-docker.pkg.dev/som-bil/ce/ce',
                    'tags': [RUNNER], #['fornix'], # sherkube # declare in main ci.yaml as global variable
                    'retry': {'max':2} # how many times it tries to resolve internal failures
                  }
                }
config_dict =  config_runner 

with open('sims-gitlab-ci.yml','w') as yml_file:
    yaml.safe_dump(config_dict, yml_file)

for j in range(num_child):
    file_num = str(j).zfill(5)  #99,999 jobs maximum 
    print(file_num)
    config_dict = {'job'+str(j):
                    {'variables': {'NUM_JOB':j, 'KUBERNETES_CPU_REQUEST':1, 'KUBERNETES_MEMORY_REQUEST': '2Gi'},  
                     'script' : ['pip3 install nengo', 'python3 frobenius_sweep.py', 'echo "Simulation finished"'], #pipeling_WIP.py
                     'needs' : [{'pipeline': '$PARENT_PIPELINE_ID', 'job':'job01'}], #, 'artifacts': True}]
                    }
                  }

    with open('sims-gitlab-ci.yml','a') as yml_file:
        yaml.safe_dump(config_dict, yml_file)