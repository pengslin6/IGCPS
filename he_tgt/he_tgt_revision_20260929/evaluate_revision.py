"""Recompute HE-TGT main tables, fixed component removals and ranking controls.

All test predictions are retrospective. No tuning, fitting or selection occurs here.
"""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'unified_detection_20260929'))
import pair_model as pm
import unified as u
sys.path.insert(0,str(ROOT/'experiments_20260928'))
import common as audit
from sklearn.metrics import log_loss
from scipy.stats import spearmanr
np,torch=u.np,u.torch


def calibration(p,y):
    confidence=p.max(1); correct=p.argmax(1)==y
    bins=[]; ece=0.
    for lo,hi in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]):
        mask=(confidence>lo)&(confidence<=hi)
        if mask.any():
            acc=float(correct[mask].mean()); conf=float(confidence[mask].mean())
            ece+=mask.mean()*abs(acc-conf)
            bins.append({'n':int(mask.sum()),'accuracy':acc,'confidence':conf})
    entropy=-(p*np.log(np.maximum(p,1e-15))).sum(1)
    return {'ece':float(ece),'brier':float(((p-np.eye(p.shape[1])[y])**2).sum(1).mean()),
        'nll':float(log_loss(y,p)),'entropy_error_spearman':float(spearmanr(entropy,~correct).statistic),
        'confidence':float(confidence.mean()*100),'accuracy':float(correct.mean()*100),'bins':bins}


def main():
    torch.set_num_threads(2)
    locked=u.read(pm.OUT/'final_fit_lock.json')
    for path,digest in locked['hashes'].items():
        assert u.sha(path)==digest,path
    report={'scope':'Retrospective fixed ablations; no new selection or refitting',
            'policy':locked['policy'],'datasets':{}}
    for d in u.DATASETS:
        graph=u.graph(d,'test'); development=u.graph(d,'development')
        arr=dict(np.load(u.MODEL/d/'test.npz'))
        head=u.joblib.load(pm.OUT/d/'final_pairs.joblib')
        prior=u.joblib.load(pm.OUT/d/'final_prior.joblib')
        pp=pm.pair_probabilities(head,arr); hp=u.probabilities(prior,arr)
        lookup=audit.c.class_root_table(development,len(u.meta(d)['label_names']))
        rows=[]
        for seed in u.SEEDS:
            base=np.load(u.MODEL/d/f'seed{seed}.predictions.npz')
            old=np.load(u.AUDIT/'archived_analysis'/d/f'replayed__seed{seed}.npz')
            saved=np.load(pm.OUT/d/f'final_seed{seed}.npz')
            for source in [base,old,saved]:
                for key in ['labels','raw_indices']:
                    np.testing.assert_array_equal(source[key],graph[key].numpy())
            g=base['base_probabilities']; roots=base['root_scores']
            full=pm.decision(g,hp,pp,locked['policy'])
            np.testing.assert_array_equal(full,saved['probabilities'])
            # Pair-only is G + pair refinement, with no multiclass history prior.
            pair=pm.decision(g,hp,pp,{**locked['policy'],'gamma':0.})
            variants={'original_DA-TGT':(old['probabilities'],old['root_scores']),
                'matched_graph':(g,roots),'history_only':(hp,roots),
                'graph_history':(.75*g+.25*hp,roots),'pair_only':(pair,roots),'HE-TGT':(full,roots)}
            row={'seed':seed,'variants':{}}
            for name,(p,s) in variants.items():
                row['variants'][name]={**u.metrics(d,arr['labels'],p.argmax(1)),**audit.c.evaluate(p,s,graph,lookup)}
            row['calibration']=calibration(full,arr['labels'])
            eligibility=graph['root_mask'].numpy()>0
            row['groups']={}
            for name,mask in audit.groups(arr['labels'],full,u.meta(d)['label_names']).items():
                selected=eligibility & mask
                if selected.any():
                    cache={k:v[torch.tensor(selected)] for k,v in graph.items() if torch.is_tensor(v) and len(v)==len(selected)}
                    row['groups'][name]={'n':int(selected.sum()),'head':audit.c.ranking(roots[selected],cache),
                        'lookup':audit.c.ranking((full@lookup)[selected],cache)}
            rows.append(row)
            if seed==11:
                out=HERE/d; out.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(out/'representative.npz',labels=arr['labels'],probabilities=full,root_scores=roots,
                    root_targets=graph['root_targets'].numpy(),raw_indices=arr['raw_indices'])
            print('VERIFIED',d,seed,round(row['variants']['HE-TGT']['f1'],4),flush=True)
        fields=['accuracy','f1','auc','top1','mrr','ndcg5','normal_fpr','anomaly_recall','focus_recall','focus_precision','focus_f1']
        summary={}
        for name in rows[0]['variants']:
            summary[name]={k:{'mean':float(np.mean([r['variants'][name][k] for r in rows])),
                'sd':float(np.std([r['variants'][name][k] for r in rows],ddof=1))} for k in fields}
        for k in ['top1','mrr','ndcg5']:
            summary['HE-TGT']['lookup_'+k]={'mean':float(np.mean([r['variants']['HE-TGT']['class_lookup_soft'][k] for r in rows])),
                'sd':float(np.std([r['variants']['HE-TGT']['class_lookup_soft'][k] for r in rows],ddof=1))}
        report['datasets'][d]={'rows':rows,'summary':summary,'metadata':u.meta(d)}
        u.dump(HERE/'metrics.json',report)
        print('SUMMARY',d,{k:(v['mean'],v['sd']) for k,v in summary['HE-TGT'].items()},flush=True)
    u.dump(HERE/'verification.json',{'full_probabilities_identical':True,'raw_indices_and_labels_identical':True,
        'source_sha256':u.sha(__file__),'metrics_sha256':u.sha(HERE/'metrics.json'),
        'selection_or_new_fits':False,'historical_final_fit_lock_sha256':u.sha(pm.OUT/'final_fit_lock.json')})


if __name__=='__main__':
    with u.threadpool_limits(limits=2):
        main()
