from transformers import FunnelConfig, AdamW, Trainer, TrainingArguments
import json
import datetime
import os

prefix=""
stamp = prefix+str(datetime.date.today())+"-"+str(datetime.datetime.now().hour)+"-"+str(datetime.datetime.now().minute)
if not os.path.exists(stamp):
    os.mkdir(stamp)
path = prefix+'config.json'
stand = prefix+'standardconfig.json'
diffpath = stamp+'/diff.json'
logconfigpath = stamp+'/logconfig.json'

with open(path,'r') as fp:
    data=json.load(fp)

for x in ['pretrain','finetune']:
    for y in ["output_dir","logging_dir","run_name"]:
        data[x][y]=stamp+"/"+data[x][y]

modelconfig=FunnelConfig(**(data['model']))
training_args_pt = TrainingArguments(**(data['pretrain']))
training_args_ft = TrainingArguments(**(data['finetune']))

with open(logconfigpath,'w') as fp:
    json.dump(data, fp, indent=4)
'''with open(stand,'r') as fp:
    standconfig=json.load(fp)

diff=dict()
for type in data.keys():
    diff[type]=dict()
    for key in data[type]:
        if key not in standconfig[type].keys() or standconfig[type].keys()!=data[type][key]:
            diff[type][key] = data[type][key]

with open(diffpath,'w') as fp:
    json.dump(diff, fp, indent=4)'''