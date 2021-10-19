#! /env/bin/python

import matplotlib.pyplot as plt
import torch 
import joblib
from pathlib import Path
from collections import defaultdict
dir_base=Path("/scratch/akitaitsev/intersubject_generalizeation/dnn/perceiver/trial2/EEG/")
losses=["leftthomas", "zablo"]
lrs=["lr0.01","lr0.05","lr0.001"]

data=defaultdict()
legend=[]
fig, ax=plt.subplots(1)
for loss in losses:
    for lr in lrs:
        data[loss+lr]=[]
        dat_tmp = joblib.load(dir_base.joinpath(loss, lr, "losses.pkl"))
        for epoch in range(100): 
            data[loss+lr] = data[loss+lr]+ dat_tmp["epoch"+str(epoch)]
        legend.append(loss+lr)
        ax.plot(data[loss+lr])
fig.legend(legend)
plt.show()
