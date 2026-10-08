"""Apply the current fixed HE-TGT history/pair rule to matched graph validation probabilities."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'ranking_comparison_20260929'))
import study as s
np,u=s.np,s.u

def main():
    out={'scope':'secondary validation diagnostic; not part of the predeclared removal gate','test_accessed':False,'datasets':{}}
    for d in s.DATASETS:
        arr=u.data(d,'validation');cache=u.graph(d,'validation');train=u.graph(d,'train')
        history=s.joblib.load(u.HERE/d/'screen/full_history__leaves7_min60.joblib')
        hp=u.probabilities(history,arr)
        pair_head=s.joblib.load(s.pair.OUT/d/'screen/network_with_physical_changes__min60.joblib')
        pp=s.pair.pair_probabilities(pair_head,arr);table=s.lookup(train,len(u.meta(d)['label_names']))
        block={}
        for kind in ['with_head','without_head']:
            rows=[]
            for seed in [11,22,33]:
                graph=np.load(HERE/d/f'{kind}_seed{seed}.npz')['probabilities']
                p=s.decision(graph,hp,pp,.75,.5)
                m=s.evaluate(d,p,cache,table);rows.append({'seed':seed,'metrics':m})
                np.savez_compressed(HERE/d/f'full_pipeline_{kind}_seed{seed}.npz',probabilities=p,labels=arr['labels'],raw_indices=arr['raw_indices'],node_scores=p@table)
            block[kind]={'rows':rows,'summary':{k:{'mean':float(np.mean([r['metrics'][k] for r in rows])),'sd':float(np.std([r['metrics'][k] for r in rows],ddof=1))} for k in ['f1','top1','normal_fpr','anomaly_recall','focus_recall','focus_f1']}}
        block['without_minus_with']={k:block['without_head']['summary'][k]['mean']-block['with_head']['summary'][k]['mean'] for k in block['with_head']['summary']}
        out['datasets'][d]=block
    u.dump(HERE/'full_pipeline_diagnostic.json',out);print({d:v['without_minus_with'] for d,v in out['datasets'].items()},flush=True)

if __name__=='__main__':
    with u.threadpool_limits(limits=2):main()
