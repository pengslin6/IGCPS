from pathlib import Path
import argparse, json, hashlib, sys, time, os, subprocess, random

SOURCE=Path(__file__).resolve().parent
HERE=SOURCE
ASSETS=SOURCE.parent
SEEDS=[11,22,33,44,55]
DATASETS=['IGCPS','TE-CUP-SEC']
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)
def setup():
    sys.dont_write_bytecode=True
    os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
    sys.path.insert(0,str(SOURCE.parent/'ranking_comparison_20260929'))
    import study as s
    s.u.ROOT=ASSETS
    s.u.MODEL=ASSETS/'detection_history_final_20260929'
    s.u.AUDIT=ASSETS/'experiments_20260928'
    s.pair.OUT=ASSETS/'unified_detection_20260929/pair_refinement'
    s.torch.set_num_threads(2)
    s.torch.set_num_interop_threads(1)
    return s
def sources():
    files=[SOURCE/'PROTOCOL.md',Path(__file__),SOURCE.parent/'ranking_comparison_20260929/study.py',SOURCE.parent/'detection_history_final_20260929/retained_runtime/backbone.py',SOURCE.parent/'unified_detection_20260929/pair_model.py']
    for ds in DATASETS:
        files += [ASSETS/'experiments_20260928/fair_control'/ds/'selection_lock.json',ASSETS/'experiments_20260928/data_cache'/ds/'development.pt',ASSETS/'experiments_20260928/data_cache'/ds/'test.pt',ASSETS/'experiments_20260928/data_cache'/ds/'adjacency.pt',ASSETS/'detection_history_final_20260929'/ds/'test.npz',ASSETS/'unified_detection_20260929/pair_refinement'/ds/'final_prior.joblib',ASSETS/'unified_detection_20260929/pair_refinement'/ds/'final_pairs.joblib']
        for seed in SEEDS:files += [ASSETS/'experiments_20260928/fair_control'/ds/f'final__DA-TGT__seed{seed}.pt',ASSETS/'experiments_20260928/fair_control'/ds/f'final__DA-TGT__seed{seed}.predictions.npz',ASSETS/'unified_detection_20260929/pair_refinement'/ds/f'final_seed{seed}.npz']
    return {str(p):sha(p) for p in files}
def prepare():
    lock=HERE/'input_lock.json'
    if lock.exists():assert read(lock)['hashes']==sources();return
    chosen={d:read(ASSETS/'experiments_20260928/fair_control'/d/'selection_lock.json')['selected']['DA-TGT'] for d in DATASETS}
    dump(lock,{'hashes':sources(),'selected_graph_settings':{d:{'learning_rate':v['learning_rate'],'epochs':v['epoch']} for d,v in chosen.items()},'seeds':SEEDS,'policy':{'gamma':.25,'beta':1.,'rho':.5},'test_used_for_new_selection':False,'execution_scope':'numerical model refit; not a device latency benchmark'})
    print('Input/code hashes and fixed settings locked.',flush=True)
def train(dataset,seed):
    s=setup();np,torch,u=s.np,s.torch,s.u
    lock=read(HERE/'input_lock.json');assert sha(Path(__file__))==lock['hashes'][str(Path(__file__))]
    cache=u.graph(dataset,'development');meta=u.meta(dataset)
    cfg=lock['selected_graph_settings'][dataset]
    out=HERE/dataset/f'no_head_seed{seed}';out.parent.mkdir(exist_ok=True)
    if out.with_suffix('.fit.json').exists():
        info=read(out.with_suffix('.fit.json'));assert sha(out.with_suffix('.pt'))==info['checkpoint_sha256'];print('Already fitted',dataset,seed,flush=True);return
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=s.NodeFreeGraph(cache['features'].shape[-1],meta)
    assert not any('root' in n for n,_ in model.named_parameters())
    adjacency=torch.load(u.AUDIT/'data_cache'/dataset/'adjacency.pt',weights_only=True)
    # Check the retained no-head forward against the same decoupled base model.
    from dataclasses import replace
    reference=u.DATGTJoint(cache['features'].shape[-1],meta['n_nodes'],meta['n_net'],len(meta['label_names']),replace(u.CANDIDATE,decouple_detection_root=True,root_weight=0.,normal_root_weight=0.))
    reference.load_state_dict(model.state_dict(),strict=False)
    reference.eval();model.eval()
    with torch.inference_mode():torch.testing.assert_close(model(cache['features'][:8],adjacency),reference(cache['features'][:8],adjacency)['anomaly_logits'],atol=1e-6,rtol=1e-6)
    del reference
    # Reset seed after the implementation check, so training randomness is reproducible.
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=s.NodeFreeGraph(cache['features'].shape[-1],meta)
    weights=torch.bincount(cache['labels'],minlength=len(meta['label_names'])).float()
    weights=torch.sqrt(weights.max()/weights.clamp_min(1));weights/=weights.min()
    loader=s.DataLoader(s.TensorDataset(cache['features'],cache['labels'],cache['recency']),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(seed))
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=.0005)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=18,eta_min=cfg['learning_rate']*.1)
    history=[];begin=time.perf_counter()
    with u.threadpool_limits(limits=2):
        for epoch in range(1,cfg['epochs']+1):
            tick=time.perf_counter();model.train();total=0.
            for x,y,recency in loader:
                opt.zero_grad(set_to_none=True)
                logits=model(x,adjacency)
                loss=(s.F.cross_entropy(logits,y,weight=weights,reduction='none')*recency).mean()
                assert torch.isfinite(loss)
                loss.backward();s.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=float(loss.detach())*len(y)
            sched.step();row={'epoch':epoch,'loss':total/len(cache['labels']),'seconds':time.perf_counter()-tick};history.append(row)
            dump(out.with_suffix('.history.json'),history)
            print('TRAIN',dataset,seed,epoch,'/',cfg['epochs'],'loss',round(row['loss'],5),'seconds',round(row['seconds'],2),flush=True)
    torch.save({'state_dict':model.state_dict(),'dataset':dataset,'seed':seed,'settings':cfg,'input_lock_sha256':sha(HERE/'input_lock.json')},out.with_suffix('.pt'))
    dump(out.with_suffix('.fit.json'),{'dataset':dataset,'seed':seed,'settings':cfg,'rows':len(cache['labels']),'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_sha256':sha(out.with_suffix('.pt')),'elapsed_fit_seconds':time.perf_counter()-begin,'test_accessed':False,'implementation_equivalence_passed':True})
def train_all():
    prepare();queue=[(d,seed) for d in ['TE-CUP-SEC','IGCPS'] for seed in SEEDS];active=[];completed=0
    while queue or active:
        while queue and len(active)<3:
            d,seed=queue.pop(0);dest=HERE/d;dest.mkdir(exist_ok=True)
            handle=(dest/f'train_seed{seed}.log').open('w',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),'train','--dataset',d,'--seed',str(seed),'--artifact-root',str(ASSETS),'--output-root',str(HERE)],stdout=handle,stderr=subprocess.STDOUT)
            active.append((d,seed,proc,handle));print('Started',d,seed,flush=True)
        for item in active[:]:
            d,seed,proc,handle=item
            if proc.poll() is not None:
                handle.close();active.remove(item)
                if proc.returncode:
                    for _,_,p,h in active:p.terminate();h.close()
                    raise RuntimeError(f'Training failed {d} seed {seed}; see {d}/train_seed{seed}.log')
                completed+=1;print('Completed fits',completed,'/10',d,seed,flush=True)
        time.sleep(1)
    dump(HERE/'all_fits_lock.json',{'input_lock_sha256':sha(HERE/'input_lock.json'),'checkpoints':{str(HERE/d/f'no_head_seed{s}.pt'):sha(HERE/d/f'no_head_seed{s}.pt') for d in DATASETS for s in SEEDS},'test_evaluated_by_this_run':False})
    print('ALL TEN FITS LOCKED before test evaluation.',flush=True)
def evaluate():
    s=setup();np,torch,u=s.np,s.torch,s.u
    lock=read(HERE/'input_lock.json');fit=read(HERE/'all_fits_lock.json')
    assert sha(HERE/'input_lock.json')==fit['input_lock_sha256']
    assert sources()==lock['hashes']
    for path,digest in fit['checkpoints'].items():assert sha(path)==digest
    results={'scope':'retrospective fixed-setting full-pipeline deletion','execution_scope':'numerical model comparison; not a device latency benchmark','with_head':'reused verified main checkpoints','without_head':'ten newly fitted development models','policy':lock['policy'],'datasets':{}}
    with u.threadpool_limits(limits=2):
        for dataset in DATASETS:
            test=u.graph(dataset,'test');dev=u.graph(dataset,'development');meta=u.meta(dataset)
            arr=dict(np.load(u.MODEL/dataset/'test.npz'))
            for key in ['labels','raw_indices']:np.testing.assert_array_equal(test[key].numpy(),arr[key])
            table=s.lookup(dev,len(meta['label_names']));adj=torch.load(u.AUDIT/'data_cache'/dataset/'adjacency.pt',weights_only=True)
            prior=u.probabilities(s.joblib.load(s.pair.OUT/dataset/'final_prior.joblib'),arr)
            pairs=s.pair.pair_probabilities(s.joblib.load(s.pair.OUT/dataset/'final_pairs.joblib'),arr)
            block={k:[] for k in ['with_head','without_head']}
            for seed in SEEDS:
                for variant in ['with_head','without_head']:
                    if variant=='with_head':
                        checkpoint=torch.load(ASSETS/'experiments_20260928/fair_control'/dataset/f'final__DA-TGT__seed{seed}.pt',map_location='cpu',weights_only=False)
                        model=u.DATGTJoint(test['features'].shape[-1],meta['n_nodes'],meta['n_net'],len(meta['label_names']),u.CANDIDATE)
                    else:
                        checkpoint=torch.load(HERE/dataset/f'no_head_seed{seed}.pt',map_location='cpu',weights_only=False)
                        model=s.NodeFreeGraph(test['features'].shape[-1],meta)
                    model.load_state_dict(checkpoint['state_dict']);model.eval();g=[];roots=[]
                    with torch.inference_mode():
                        for x in test['features'].split(128):
                            out=model(x,adj)
                            if variant=='with_head':g.append(out['anomaly_logits'].softmax(-1).numpy());roots.append(out['root_logits'].sigmoid().numpy())
                            else:g.append(out.softmax(-1).numpy())
                    g=np.concatenate(g);p=s.pair.decision(g,prior,pairs,lock['policy'])
                    if variant=='with_head':
                        with np.load(ASSETS/'experiments_20260928/fair_control'/dataset/f'final__DA-TGT__seed{seed}.predictions.npz') as old:
                            np.testing.assert_allclose(g,old['probabilities'],atol=1e-6,rtol=1e-6)
                            np.testing.assert_allclose(np.concatenate(roots),old['root_scores'],atol=1e-6,rtol=1e-6)
                        with np.load(s.pair.OUT/dataset/f'final_seed{seed}.npz') as old:np.testing.assert_allclose(p,old['probabilities'],atol=1e-6,rtol=1e-6)
                    m=s.evaluate(dataset,p,test,table,auc=True)
                    if roots:m['direct_node']=s.rank(np.concatenate(roots),test)
                    row={'dataset':dataset,'variant':variant,'seed':seed,'metrics':m,'parameters':sum(q.numel() for q in model.parameters())};block[variant].append(row)
                    path=HERE/dataset/f'{variant}_seed{seed}.predictions.npz'
                    np.savez_compressed(path,labels=arr['labels'],raw_indices=arr['raw_indices'],probabilities=p,graph_probabilities=g,lookup_scores=p@table)
                    dump(HERE/dataset/f'{variant}_seed{seed}.metrics.json',row)
                    print('EVAL',dataset,variant,seed,'F1',round(m['f1'],3),'focus recall',round(m['focus_recall'],3),flush=True)
                    del model
            fields=['accuracy','f1','auc','normal_fpr','anomaly_recall','focus_recall','focus_precision','focus_f1','top1','mrr','ndcg5']
            summary={variant:{k:{'mean':float(np.mean([r['metrics'][k] for r in rows])),'sd':float(np.std([r['metrics'][k] for r in rows],ddof=1))} for k in fields} for variant,rows in block.items()}
            differences={k:[float(b['metrics'][k]-a['metrics'][k]) for a,b in zip(block['with_head'],block['without_head'])] for k in fields}
            results['datasets'][dataset]={'rows':block,'summary':summary,'without_minus_with':{k:{'mean':float(np.mean(v)),'sd':float(np.std(v,ddof=1)),'seed_differences':v} for k,v in differences.items()}}
    dump(HERE/'results.json',results)
    lines=['# No-head HE-TGT: five-seed full test comparison','','Retrospective fixed settings; ten new no-head fits. With-head main checkpoints replayed. Numerical comparison; no inference-latency measurement.','']
    for dataset,block in results['datasets'].items():
        lines+=['## '+dataset,'','| Variant | Macro F1 | Normal FPR ↓ | Focus recall | Focus F1 | Lookup Top-1 |','|---|---:|---:|---:|---:|---:|']
        for variant in ['with_head','without_head']:
            q=block['summary'][variant];lines.append('|'+variant+'|'+'|'.join(f"{q[k]['mean']:.3f} ± {q[k]['sd']:.3f}" for k in ['f1','normal_fpr','focus_recall','focus_f1','top1'])+'|')
        lines+=['', 'No-head minus with-head paired macro-F1 changes: '+str(block['without_minus_with']['f1']['seed_differences']),'']
    (HERE/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Complete results saved',flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','train','train_all','evaluate']);parser.add_argument('--artifact-root',type=Path,required=True);parser.add_argument('--output-root',type=Path,required=True);parser.add_argument('--dataset',choices=DATASETS);parser.add_argument('--seed',type=int,choices=SEEDS);a=parser.parse_args()
    ASSETS=a.artifact_root.resolve();HERE=a.output_root.resolve();HERE.mkdir(parents=True,exist_ok=True)
    if a.phase=='train':train(a.dataset,a.seed)
    else:globals()[a.phase]()
