from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import sys,time,random,json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'ranking_comparison_20260929'))
import study as s
np,torch,u=s.np,s.torch,s.u
from torch.utils.data import DataLoader,TensorDataset
from torch.nn import functional as F
SEEDS=[11,22,33];DATASETS=s.DATASETS

def root_loss(logits,targets,eligible,labels):
    node_weights=torch.ones_like(targets);counts=targets.sum(1,keepdim=True).clamp_min(1.)
    node_weights=torch.where(targets>0,torch.maximum(torch.full_like(targets,5.),torch.full_like(targets,targets.shape[1])/counts),node_weights)
    row=F.binary_cross_entropy_with_logits(logits,targets,weight=node_weights,reduction='none').mean(1)
    positive=(row*eligible).sum()/eligible.sum().clamp_min(1.)
    normal=(labels==0)
    negative=F.binary_cross_entropy_with_logits(logits[normal],targets[normal]) if normal.any() else logits.sum()*0
    return positive+.03*negative

def rank(scores,cache):return s.rank(scores,cache)

def metrics(d,p,cache,table,scores=None):
    out=s.evaluate(d,p,cache,table)
    if scores is not None:out['direct_node']=rank(scores,cache)
    return out

def build(d,cache,kind):
    if kind=='without_head':return s.NodeFreeGraph(cache['features'].shape[-1],u.meta(d))
    return u.DATGTJoint(cache['features'].shape[-1],u.meta(d)['n_nodes'],u.meta(d)['n_net'],len(u.meta(d)['label_names']),u.CANDIDATE)

def fit_one(args):
    d,kind,seed=args;torch.set_num_threads(2);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    train=u.graph(d,'train');val=u.graph(d,'validation');meta=u.meta(d);model=build(d,train,kind)
    adj=torch.load(u.AUDIT/'data_cache'/d/'adjacency.pt',weights_only=True)
    counts=torch.bincount(train['labels'],minlength=len(meta['label_names'])).float();cw=torch.sqrt(counts.max()/counts.clamp_min(1));cw/=cw.min()
    loader=DataLoader(TensorDataset(train['features'],train['labels'],train['root_targets'],train['root_mask'],train['recency']),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed))
    opt=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.0005);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,18,eta_min=.00007)
    history=[]
    for epoch in range(1,19):
        start=time.perf_counter();model.train();total=0.
        for x,y,roots,eligible,recency in loader:
            opt.zero_grad(set_to_none=True);out=model(x,adj)
            logits=out if kind=='without_head' else out['anomaly_logits']
            loss=(F.cross_entropy(logits,y,weight=cw,reduction='none')*recency).mean()
            if kind=='with_head':loss=loss+.30*root_loss(out['root_logits'],roots,eligible,y)
            assert torch.isfinite(loss);loss.backward();s.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=float(loss.detach())*len(y)
        sched.step();history.append({'epoch':epoch,'loss':total/len(train['labels']),'seconds':time.perf_counter()-start})
    model.eval();probs=[];roots=[]
    with torch.inference_mode():
        for x in val['features'].split(128):
            out=model(x,adj)
            if kind=='without_head':probs.append(out.softmax(-1).numpy())
            else:probs.append(out['anomaly_logits'].softmax(-1).numpy());roots.append(out['root_logits'].sigmoid().numpy())
    p=np.concatenate(probs);r=np.concatenate(roots) if roots else None;table=s.lookup(train,len(meta['label_names']))
    m=metrics(d,p,val,table,r);folder=HERE/d;folder.mkdir(parents=True,exist_ok=True)
    prefix=folder/f'{kind}_seed{seed}';torch.save(model.state_dict(),str(prefix)+'.pt')
    np.savez_compressed(str(prefix)+'.npz',probabilities=p,node_scores=r if r is not None else p@table,labels=val['labels'].numpy(),raw_indices=val['raw_indices'].numpy())
    u.dump(str(prefix)+'.json',{'dataset':d,'kind':kind,'seed':seed,'metrics':m,'history':history,'parameters':sum(x.numel() for x in model.parameters()),'has_node_head':any('root_scorer' in n for n,_ in model.named_parameters())})
    return d,kind,seed,m

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def summarize():
    result={'protocol_sha256':sha(HERE/'PROTOCOL.md'),'source_sha256':sha(__file__),'test_accessed':False,'datasets':{}}
    for d in DATASETS:
        block={}
        for kind in ['with_head','without_head']:
            rows=[u.read(HERE/d/f'{kind}_seed{seed}.json') for seed in SEEDS]
            keys=['accuracy','f1','normal_fpr','anomaly_recall','focus_recall','focus_f1','top1','mrr','ndcg5']
            block[kind]={'seeds':rows,'summary':{k:{'mean':float(np.mean([x['metrics'][k] for x in rows])),'sd':float(np.std([x['metrics'][k] for x in rows],ddof=1))} for k in keys}}
            if kind=='with_head':block[kind]['direct_node_summary']={k:{'mean':float(np.mean([x['metrics']['direct_node'][k] for x in rows])),'sd':float(np.std([x['metrics']['direct_node'][k] for x in rows],ddof=1))} for k in ['top1','mrr','ndcg5']}
        a=block['with_head']['summary'];b=block['without_head']['summary'];diff={k:b[k]['mean']-a[k]['mean'] for k in a}
        seed_diffs=[block['without_head']['seeds'][i]['metrics']['f1']-block['with_head']['seeds'][i]['metrics']['f1'] for i in range(3)]
        checks={'f1':diff['f1']>=-.25,'normal_fpr':diff['normal_fpr']<=.5,'anomaly_recall':diff['anomaly_recall']>=-.5,'focus_recall':diff['focus_recall']>=-2.,'lookup_top1':diff['top1']>=-.1,'seed_f1':min(seed_diffs)>=-1.}
        block['without_minus_with']=diff;block['seed_f1_differences']=seed_diffs;block['checks']=checks;block['passes']=all(checks.values());result['datasets'][d]=block
    result['remove_node_head']=all(v['passes'] for v in result['datasets'].values());u.dump(HERE/'results.json',result)
    lines=['# Matched node-head removal validation','',f"Decision: {'remove' if result['remove_node_head'] else 'retain'} the node head under the predeclared rule.",'','Validation only; three matched seeds; no test access. Lookup ranking is class-associated reference ranking, not causal localization.','']
    for d,v in result['datasets'].items():
        lines+=['## '+d,'','|Variant|Macro F1|Lookup Top1|Normal FPR|Anomaly recall|Focus recall|','|---|---:|---:|---:|---:|---:|']
        for kind in ['with_head','without_head']:
            q=v[kind]['summary'];lines.append('|'+kind+'|'+'|'.join(f"{q[k]['mean']:.3f}±{q[k]['sd']:.3f}" for k in ['f1','top1','normal_fpr','anomaly_recall','focus_recall'])+'|')
        q=v['with_head']['direct_node_summary'];lines+=['',f"With-head direct node Top1: {q['top1']['mean']:.3f}±{q['top1']['sd']:.3f}.",f"Without-minus-with differences: {json.dumps(v['without_minus_with'],ensure_ascii=False)}",f"Checks: {json.dumps(v['checks'],ensure_ascii=False)}",'']
    (HERE/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps({'remove_node_head':result['remove_node_head'],'passes':{d:v['passes'] for d,v in result['datasets'].items()}},ensure_ascii=False),flush=True)

if __name__=='__main__':
    tasks=[(d,k,seed) for d in DATASETS for k in ['with_head','without_head'] for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=3) as pool:
        fs=[pool.submit(fit_one,t) for t in tasks]
        for f in as_completed(fs):print('FINISHED',f.result()[:3],flush=True)
    summarize()
