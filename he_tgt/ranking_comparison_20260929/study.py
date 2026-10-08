"""Locked development comparison: no-node-head graph/history/lookup variants."""
from pathlib import Path
from dataclasses import replace
import argparse
import gc
import itertools
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'unified_detection_20260929'))
import unified as u
import pair_model as pair
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
np,torch,joblib=u.np,u.torch,u.joblib
DATASETS=u.DATASETS
BASE={'view':'full_history','leaves':7,'minimum':60,'iterations':200}
FIELDS=u.KEYS+['top1','mrr','ndcg5']


def signature(s):
    return f"{s['view']}__l{s['leaves']}_m{s['minimum']}_i{s['iterations']}"


def settings():
    return [dict(view=v,leaves=l,minimum=m,iterations=i)
            for v,l,m,i in itertools.product(['full_history','network_with_physical_changes'],[7,15],[20,60],[200,400])]


def lookup(cache,classes):
    y=cache['labels'].numpy(); r=cache['root_targets'].numpy()
    eligible=cache['root_mask'].numpy()>0
    table=np.zeros((classes,r.shape[1]),np.float32)
    for c in range(classes):
        mask=(y==c)&eligible
        if mask.any():
            table[c]=r[mask].mean(0)
    return table


def rank(scores,cache,mask=None):
    eligible=cache['root_mask'].numpy()>0
    if mask is not None: eligible &= mask
    truth=cache['root_targets'].numpy()[eligible]
    scores=scores[eligible]
    order=np.argsort(scores,axis=1)[:,::-1]
    correct=np.take_along_axis(truth,order,axis=1)
    first=np.argmax(correct>0,axis=1)+1
    discount=1/np.log2(np.arange(2,min(5,truth.shape[1])+2))
    ideal=np.array([discount[:min(int(row.sum()),5)].sum() for row in truth])
    return {'top1':float(np.mean(first==1)*100),'mrr':float(np.mean(1/first)*100),
            'ndcg5':float(np.mean((correct[:,:len(discount)]*discount).sum(1)/ideal)*100)}


def evaluate(d,p,cache,table,mask=None,auc=False):
    y=cache['labels'].numpy()
    if mask is None: mask=np.ones(len(y),bool)
    out={**u.metrics(d,y[mask],p[mask].argmax(1)),**rank(p@table,cache,mask)}
    if auc:
        p64=p.astype(np.float64);p64/=p64.sum(1,keepdims=True)
        out['auc']=float(roc_auc_score(y[mask],p64[mask],multi_class='ovr',average='macro')*100)
    assert abs(out['f1']-100*f1_score(y[mask],p[mask].argmax(1),average='macro'))<1e-8
    return out


def halves(cache):
    y=cache['labels'].numpy(); raw=cache['raw_indices'].numpy()
    first=np.zeros(len(y),bool)
    for c in np.unique(y):
        ids=np.flatnonzero(y==c);ids=ids[np.argsort(raw[ids],kind='stable')]
        first[ids[:len(ids)//2]]=True
    return [first,~first]


class NodeFreeGraph(u.DATGTJoint):
    def __init__(self,dim,meta):
        super().__init__(dim,meta['n_nodes'],meta['n_net'],len(meta['label_names']),
                         replace(u.CANDIDATE,decouple_detection_root=True,root_weight=0.,normal_root_weight=0.))
        del self.root_scorer
        assert not any('root' in name for name,_ in self.named_parameters())

    def forward(self,features,adjacency):
        if self.training and self.input_noise>0:
            features=features+torch.randn_like(features)*self.input_noise
        hidden=self.input_projection(features)+self.position_embedding
        net=hidden[:,:self.n_net]+self.type_embedding[0]
        phy=hidden[:,self.n_net:]+self.type_embedding[1]
        hidden=self.adapter_norm(torch.cat([net+self.net_adapter(net),phy+self.phy_adapter(phy)],1))
        for block in self.local_blocks: hidden=block(hidden,adjacency)
        encoded=self.global_encoder(torch.cat([self.graph_token.expand(len(hidden),-1,-1),hidden],1))
        graph,nodes=encoded[:,0],encoded[:,1:]
        net,phy=nodes[:,:self.n_net],nodes[:,self.n_net:]
        net_pool=self._attention_pool(net,self.net_pool)
        phy_pool=self._attention_pool(phy,self.phy_pool)
        source=self._attention_pool(nodes,self.detection_pool)
        representation=torch.cat([graph,net_pool,phy_pool,net.max(1).values,phy.max(1).values,
                                  source,torch.abs(net_pool-phy_pool)],-1)
        return self.classifier(self.detection_norm(representation))


def graph_predict(model,cache,adjacency):
    model.eval()
    with torch.inference_mode():
        return np.concatenate([model(x,adjacency).softmax(-1).numpy() for x in cache['features'].split(128)])


def fit_graph(d,cache,lr,seed,epochs,out,validation=None):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    meta=u.meta(d); model=NodeFreeGraph(cache['features'].shape[-1],meta)
    adjacency=torch.load(u.AUDIT/'data_cache'/d/'adjacency.pt',weights_only=True)
    counts=torch.bincount(cache['labels'],minlength=len(meta['label_names'])).float()
    weights=torch.sqrt(counts.max()/counts.clamp_min(1))
    # Match the retained class-weight normalization.
    weights=weights/weights.min()
    loader=DataLoader(TensorDataset(cache['features'],cache['labels'],cache['recency']),
                      batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed))
    optim=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0005)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=18,eta_min=lr*.1)
    table=lookup(cache,len(meta['label_names']))
    history=[]
    for epoch in range(1,epochs+1):
        begin=time.perf_counter();model.train();loss_sum=0.
        for x,y,recency in loader:
            optim.zero_grad(set_to_none=True)
            logits=model(x,adjacency)
            loss=(F.cross_entropy(logits,y,weight=weights,reduction='none')*recency).mean()
            assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
            optim.step();loss_sum+=float(loss.detach())*len(y)
        scheduler.step()
        record={'epoch':epoch,'loss':loss_sum/len(cache['labels']),'seconds':time.perf_counter()-begin}
        if validation is not None and epoch in [6,12,18]:
            p=graph_predict(model,validation,adjacency)
            record['metrics']=evaluate(d,p,validation,table)
            prefix=out/f'lr{int(lr*1e6)}_e{epoch}'
            np.savez_compressed(str(prefix)+'.npz',probabilities=p,labels=validation['labels'].numpy(),
                                raw_indices=validation['raw_indices'].numpy())
            torch.save(model.state_dict(),str(prefix)+'.pt')
        history.append(record);u.dump(out/f'lr{int(lr*1e6)}_seed{seed}.history.json',history)
        print('GRAPH',d,seed,lr,epoch,round(record['loss'],5),round(record['seconds'],1),
              record.get('metrics',{}).get('f1',''),flush=True)
    return model


def train_graph(d):
    out=HERE/d/'graph_validation';out.mkdir(parents=True,exist_ok=True)
    train,val=u.graph(d,'train'),u.graph(d,'validation')
    for lr in [.0007,.002]:
        if (out/f'lr{int(lr*1e6)}_e18.npz').exists():continue
        fit_graph(d,train,lr,42,18,out,val)
        gc.collect()


def fit_tree(arr,schema,s):
    x=u.features(arr,schema,s);scaler=StandardScaler().fit(x)
    y=arr['labels']; weights=np.empty(len(y))
    for c in range(schema['classes']):
        ids=np.flatnonzero(y==c);ids=ids[np.argsort(arr['raw_indices'][ids],kind='stable')]
        weights[ids]=len(y)/(schema['classes']*len(ids))*np.linspace(1,2,len(ids))
    weights/=weights.mean()
    model=HistGradientBoostingClassifier(max_iter=s['iterations'],learning_rate=.05,max_leaf_nodes=s['leaves'],
                min_samples_leaf=s['minimum'],l2_regularization=5,max_bins=128,early_stopping=False,random_state=0)
    model.fit(np.clip(scaler.transform(x),-10,10),y,sample_weight=weights)
    return {'schema':schema,'setting':s,'model':model,'scaler':scaler}


def train_trees(d):
    out=HERE/d/'trees_validation';out.mkdir(parents=True,exist_ok=True)
    tr,val=u.data(d,'train'),u.data(d,'validation');schema=u.feature_metadata(d)
    for s in settings():
        name=signature(s);dest=out/(name+'.joblib')
        if dest.exists():continue
        begin=time.perf_counter()
        old=u.HERE/d/'screen'/f"{s['view']}__leaves{s['leaves']}_min{s['minimum']}.joblib"
        if s['iterations']==200 and old.exists():
            head=joblib.load(old);head['setting']={**head['setting'],**s}
            origin={'reused_training_model':str(old),'sha256':u.sha(old)}
        else:
            head=fit_tree(tr,schema,s);origin={'new_fit':True}
        p=u.probabilities(head,val)
        joblib.dump(head,dest,compress=3)
        np.savez_compressed(out/(name+'.npz'),probabilities=p,labels=val['labels'],raw_indices=val['raw_indices'])
        u.dump(out/(name+'.json'),origin)
        print('TREE',d,name,round(time.perf_counter()-begin,2),flush=True)


def decision(g,h,pp,weight,rho):
    return pair.decision(g,h,pp,{'gamma':1-weight,'rho':rho,'beta':1.})


def failures(m,b,halves_m=None,halves_b=None):
    bad=[]
    for key,limit in [('f1',.25),('top1',.10),('accuracy',-.25),('anomaly_recall',-.5),
                      ('focus_recall',-2.),('focus_f1',-1.)]:
        if key=='top1':limit=min(limit,100.-b[key])
        if m[key]-b[key]<limit-1e-8:bad.append(key)
    if m['normal_fpr']>b['normal_fpr']+.5+1e-8:bad.append('normal_fpr')
    if halves_m is not None:
        for i,(a,z) in enumerate(zip(halves_m,halves_b)):
            for key in ['f1','top1']:
                if a[key]<z[key]-1.-1e-8:bad.append(f'half{i+1}_{key}')
            if a['normal_fpr']>z['normal_fpr']+2.+1e-8:bad.append(f'half{i+1}_normal_fpr')
    return bad


def select():
    caches={d:u.graph(d,'validation') for d in DATASETS}
    tables={d:lookup(u.graph(d,'train'),len(u.meta(d)['label_names'])) for d in DATASETS}
    graph_candidates=[]
    for lr,epoch in itertools.product([.0007,.002],[6,12,18]):
        scores={}
        for d in DATASETS:
            p=np.load(HERE/d/'graph_validation'/f'lr{int(lr*1e6)}_e{epoch}.npz')['probabilities']
            scores[d]=evaluate(d,p,caches[d],tables[d])
        four=[scores[d][k] for d in DATASETS for k in ['f1','top1']]
        graph_candidates.append({'lr':lr,'epoch':epoch,'score':4/sum(1/max(x,1e-9) for x in four),'metrics':scores})
    graph_best=max(graph_candidates,key=lambda x:(x['score'],-x['epoch'],-x['lr']))
    arrays={d:u.data(d,'validation') for d in DATASETS}
    gp={d:np.load(HERE/d/'graph_validation'/f"lr{int(graph_best['lr']*1e6)}_e{graph_best['epoch']}.npz")['probabilities'] for d in DATASETS}
    pairs={d:pair.pair_probabilities(joblib.load(pair.OUT/d/'screen/network_with_physical_changes__min60.joblib'),arrays[d]) for d in DATASETS}
    treep={d:{signature(s):np.load(HERE/d/'trees_validation'/(signature(s)+'.npz'))['probabilities'] for s in settings()} for d in DATASETS}
    masks={d:halves(caches[d]) for d in DATASETS}
    baseline={d:evaluate(d,treep[d][signature(BASE)],caches[d],tables[d]) for d in DATASETS}
    base_half={d:[evaluate(d,treep[d][signature(BASE)],caches[d],tables[d],mask) for mask in masks[d]] for d in DATASETS}
    candidates=[]
    for s in settings():
        for w,rho in itertools.product([0.,.1,.25,.5,1.],[0.,.25,.5]):
            if w==1 and s!=BASE:continue
            row={'tree':s,'graph_weight':w,'rho':rho,'metrics':{},'halves':{},'failures':[]}
            gains=[]
            for d in DATASETS:
                p=decision(gp[d],treep[d][signature(s)],pairs[d],w,rho)
                row['metrics'][d]=evaluate(d,p,caches[d],tables[d])
                row['halves'][d]=[evaluate(d,p,caches[d],tables[d],mask) for mask in masks[d]]
                row['failures'] += [d+':'+x for x in failures(row['metrics'][d],baseline[d],row['halves'][d],base_half[d])]
                gains += [row['metrics'][d][k]-baseline[d][k] for k in ['f1','top1']]
            row['minimum_gain']=min(gains);row['mean_gain']=float(np.mean(gains));candidates.append(row)
    def key(r):
        s=r['tree']
        return (-len(r['failures']),r['minimum_gain'],r['mean_gain'],-r['graph_weight'],-r['rho'],
                -s['iterations'],-s['leaves'],s['minimum'],s['view'])
    eligible=[x for x in candidates if not x['failures']]
    # Exclude the unchanged baseline when choosing a diagnostic candidate.
    changed=[x for x in candidates if x['tree']!=BASE or x['graph_weight'] or x['rho']]
    selected=max(eligible or changed,key=key)
    simple=max([x for x in candidates if not x['graph_weight'] and not x['rho']],key=key)
    lock={'graph_selection':graph_best,'graph_candidates':graph_candidates,'baseline':baseline,
          'selected':selected,'best_simple':simple,'eligible_count':len(eligible),'candidates':candidates,
          'test_used_for_selection':False,'retrospective':True,
          'source_sha256':u.sha(__file__),'protocol_sha256':u.sha(HERE/'PROTOCOL.md'),
          'locked_at':time.strftime('%Y-%m-%dT%H:%M:%S')}
    u.dump(HERE/'selection_lock.json',lock)
    print('SELECTED',json.dumps({k:lock[k] for k in ['eligible_count','selected','best_simple']},ensure_ascii=False),flush=True)


def fit_final_graph_task(task):
    d,lr,seed,epochs=task
    torch.set_num_threads(2)
    with u.threadpool_limits(limits=2):
        out=HERE/d/'final'
        path=out/f'nodefree_seed{seed}.pt'
        if not path.exists():
            development=u.graph(d,'development')
            model=fit_graph(d,development,lr,seed,epochs,out)
            torch.save(model.state_dict(),path)
        return str(path),u.sha(path)


def fit_final():
    lock=u.read(HERE/'selection_lock.json')
    assert lock['source_sha256']==u.sha(__file__)
    required={signature(x['tree']):x['tree'] for x in [lock['selected'],lock['best_simple']]}
    graph_needed=lock['selected']['graph_weight']>0
    hashes={}; tasks=[]
    for d in DATASETS:
        out=HERE/d/'final';out.mkdir(exist_ok=True)
        arr=dict(np.load(u.MODEL/d/'train.npz'))
        development=u.graph(d,'development')
        np.testing.assert_array_equal(arr['labels'],development['labels'].numpy())
        np.testing.assert_array_equal(arr['raw_indices'],development['raw_indices'].numpy())
        for name,s in required.items():
            path=out/(name+'.joblib')
            if not path.exists():joblib.dump(fit_tree(arr,u.feature_metadata(d),s),path,compress=3)
            hashes[str(path)]=u.sha(path)
        if graph_needed:
            gs=lock['graph_selection']
            for seed in u.SEEDS:
                tasks.append((d,gs['lr'],seed,gs['epoch']))
        for p in [pair.OUT/d/'final_pairs.joblib',pair.OUT/d/'final_prior.joblib']:
            hashes[str(p)]=u.sha(p)
    if tasks:
        # Separate spawned processes keep per-fit RNGs isolated on Windows.
        with ProcessPoolExecutor(max_workers=3) as executor:
            for path,digest in executor.map(fit_final_graph_task,tasks):
                hashes[path]=digest
    u.dump(HERE/'final_fit_lock.json',{'hashes':hashes,'selection_sha256':u.sha(HERE/'selection_lock.json'),
               'source_sha256':u.sha(__file__),'test_loaded':False,'graph_fitted':graph_needed})
    print('ALL FINAL WEIGHTS LOCKED',flush=True)


def final_test():
    locked=u.read(HERE/'final_fit_lock.json');lock=u.read(HERE/'selection_lock.json')
    assert locked['selection_sha256']==u.sha(HERE/'selection_lock.json')
    assert locked['source_sha256']==u.sha(__file__)
    for p,h in locked['hashes'].items():assert u.sha(p)==h,p
    report={'retrospective':True,'selection_sha256':u.sha(HERE/'selection_lock.json'),'datasets':{}}
    chosen=lock['selected']
    for d in DATASETS:
        out=HERE/d/'final';arr=dict(np.load(u.MODEL/d/'test.npz'))
        cache=u.graph(d,'test');dev=u.graph(d,'development')
        np.testing.assert_array_equal(arr['raw_indices'],cache['raw_indices'].numpy())
        np.testing.assert_array_equal(arr['labels'],cache['labels'].numpy())
        table=lookup(dev,len(u.meta(d)['label_names']))
        pp=pair.pair_probabilities(joblib.load(pair.OUT/d/'final_pairs.joblib'),arr)
        prior=u.probabilities(joblib.load(pair.OUT/d/'final_prior.joblib'),arr)
        hp=u.probabilities(joblib.load(out/(signature(chosen['tree'])+'.joblib')),arr)
        sp=u.probabilities(joblib.load(out/(signature(lock['best_simple']['tree'])+'.joblib')),arr)
        rows=[]
        seeds=u.SEEDS if locked['graph_fitted'] else [0]
        for seed in seeds:
            if locked['graph_fitted']:
                model=NodeFreeGraph(cache['features'].shape[-1],u.meta(d))
                model.load_state_dict(torch.load(out/f'nodefree_seed{seed}.pt',weights_only=True))
                gp=graph_predict(model,cache,torch.load(u.AUDIT/'data_cache'/d/'adjacency.pt',weights_only=True))
            else:gp=np.zeros_like(hp)
            variants={'original_history_lookup':prior,'best_simple_lookup':sp,
                      'selected_history_lookup':hp,
                      'selected_without_pair':(1-chosen['graph_weight'])*hp+chosen['graph_weight']*gp,
                      'selected_without_graph':decision(gp,hp,pp,0.,chosen['rho']),
                      'selected_full':decision(gp,hp,pp,chosen['graph_weight'],chosen['rho'])}
            if locked['graph_fitted']:variants['nodefree_graph_lookup']=gp
            result={}
            for name,p in variants.items():
                result[name]=evaluate(d,p,cache,table,auc=True)
                np.savez_compressed(out/f'{name}_seed{seed}.npz',probabilities=p,labels=arr['labels'],
                                    raw_indices=arr['raw_indices'],node_scores=p@table)
            rows.append({'seed':seed,'metrics':result})
            print('TEST',d,seed,{n:{k:round(v[k],3) for k in ['f1','top1','normal_fpr','focus_recall']} for n,v in result.items()},flush=True)
        summaries={name:{k:{'mean':float(np.mean([r['metrics'][name][k] for r in rows])),
                             'sd':float(np.std([r['metrics'][name][k] for r in rows],ddof=1)) if len(rows)>1 else None}
                        for k in FIELDS+['auc']} for name in rows[0]['metrics']}
        a={k:v['mean'] for k,v in summaries['selected_full'].items()}
        b={k:v['mean'] for k,v in summaries['original_history_lookup'].items()}
        bad=failures(a,b)
        comparisons={}
        for name,stats in summaries.items():
            if name=='selected_full':continue
            diffs={k:a[k]-stats[k]['mean'] for k in ['f1','top1']}
            comparisons[name]={'gains':diffs,'strictly_better_both':all(v>1e-8 for v in diffs.values())}
        report['datasets'][d]={'rows':rows,'summary':summaries,'strict_failures_vs_original_history':bad,
                               'component_comparisons':comparisons}
        u.dump(HERE/'final_results.json',report)
    report['passed_strict_development']=lock['eligible_count']>0
    report['passed_all_final_guards']=all(not x['strict_failures_vs_original_history'] for x in report['datasets'].values())
    report['strictly_above_all_components']=all(all(v['strictly_better_both'] for v in x['component_comparisons'].values()) for x in report['datasets'].values())
    u.dump(HERE/'final_results.json',report)
    print('FINAL FLAGS',{k:v for k,v in report.items() if k not in ['datasets']},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['graph','trees','select','fit','test'])
    parser.add_argument('--dataset',choices=DATASETS)
    args=parser.parse_args();torch.set_num_threads(2)
    with u.threadpool_limits(limits=2):
        if args.phase=='graph':train_graph(args.dataset)
        elif args.phase=='trees':train_trees(args.dataset)
        elif args.phase=='select':select()
        elif args.phase=='fit':fit_final()
        else:final_test()
