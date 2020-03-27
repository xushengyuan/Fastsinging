import torch

source=torch.load('checkpoint_00050000.pth',map_location=torch.device('cpu'))
source_model=source['model']
model={}
for key in source_model.keys():
    if key[:6]=='module':
        model[key[7:]]=source_model[key]
    else:
        model[key]=source_model[key]

torch.save(model,'checkpoint.pth')