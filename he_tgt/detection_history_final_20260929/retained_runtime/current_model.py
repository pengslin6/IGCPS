"""Independent inference for the accepted model; no discarded experiment imports."""
from pathlib import Path
import json
import argparse
import numpy as np
import torch
import joblib
from backbone import DATGTJoint,CANDIDATE
import history_head

HERE=Path(__file__).resolve().parent
MODEL=HERE.parent
ASSETS=HERE/'assets'
SEEDS=[11,22,33,44,55]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_backbone(dataset,seed,input_width):
    meta=read(ASSETS/dataset/'metadata.json')
    model=DATGTJoint(input_width,meta['n_nodes'],meta['n_net'],len(meta['label_names']),CANDIDATE)
    state=torch.load(ASSETS/dataset/f'backbone_seed{seed}.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(state['state_dict'],strict=True)
    return model.eval()


def ig_pair(features,probabilities):
    head=dict(np.load(ASSETS/'IGCPS/nm_pm_head.npz'))
    active=(features[:,:36,4].abs()>.5).to(features.dtype)
    network=(features[:,:36,6:]*active.unsqueeze(-1)).sum(1)/active.sum(1,keepdim=True).clamp_min(1)
    values=torch.cat([network,features[:,36:,0]],dim=-1)
    x=(values[:,head['columns']].double()-torch.from_numpy(head['mean']))/torch.from_numpy(head['scale'])
    logits=x@torch.from_numpy(head['coef']).T+torch.from_numpy(head['intercept'])
    conditional=torch.cat([torch.zeros_like(logits),logits],dim=1).softmax(1).numpy()
    pred=probabilities.argmax(1)
    mask=np.isin(pred,[2,3])
    pred[mask]=np.where(conditional[mask,0]>=.3,2,3)
    return pred


def apply_policy(dataset,original,conditional):
    pred=original.copy()
    if dataset=='IGCPS':
        mask=np.isin(pred,[0,2])
        pred[mask]=np.where(conditional[mask,1]>=.5,2,0)
    else:
        pred[(original==0)&(conditional[:,1]>=.65)]=6
        pred[(original==6)&(conditional[:,1]<.35)]=0
    return pred


def predict(dataset,seed,features,history):
    assert dataset in ['IGCPS','TE-CUP-SEC'] and seed in SEEDS
    assert len(features)>0 and len(features)==len(history['history'])==len(history['static'])
    model=load_backbone(dataset,seed,features.shape[-1])
    adjacency=torch.load(ASSETS/dataset/'adjacency.pt',weights_only=True)
    probs,roots=[],[]
    with torch.inference_mode():
        for batch in features.split(128):
            output=model(batch,adjacency)
            probs.append(output['anomaly_logits'].softmax(-1).numpy())
            roots.append(output['root_logits'].sigmoid().numpy())
    probs,roots=np.concatenate(probs),np.concatenate(roots)
    inc=ig_pair(features,probs) if dataset=='IGCPS' else probs.argmax(1)
    head=joblib.load(MODEL/dataset/f'head_seed{0 if dataset=="IGCPS" else seed}.joblib')
    conditional=history_head.predict(head,history)
    return {'predictions':apply_policy(dataset,inc,conditional),'conditional_probabilities':conditional,
        'incumbent_predictions':inc,'base_probabilities':probs,'root_scores':roots}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True,choices=['IGCPS','TE-CUP-SEC'])
    parser.add_argument('--seed',type=int,default=11,choices=SEEDS)
    parser.add_argument('--graph',required=True,type=Path)
    parser.add_argument('--history',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    assert not args.output.exists(),'Refusing to overwrite output'
    torch.set_num_threads(2)
    source=torch.load(args.graph,map_location='cpu',weights_only=True)
    graph=source['features'] if isinstance(source,dict) else source
    with np.load(args.history) as src:
        history={key:src[key] for key in ['static','history']}
    np.savez_compressed(args.output,**predict(args.dataset,args.seed,graph,history))
