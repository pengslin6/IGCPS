from pathlib import Path
import sys,json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'ranking_comparison_20260929'))
import study as s
from sklearn.metrics import f1_score,confusion_matrix
np,torch,u=s.np,s.torch,s.u

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    result=u.read(HERE/'results.json');assert result['source_sha256']==sha(HERE/'run.py') and result['protocol_sha256']==sha(HERE/'PROTOCOL.md')
    checked=0;details={}
    for d in s.DATASETS:
        train=u.graph(d,'train');val=u.graph(d,'validation');table=s.lookup(train,len(u.meta(d)['label_names']))
        for kind in ['with_head','without_head']:
            for seed in [11,22,33]:
                record=u.read(HERE/d/f'{kind}_seed{seed}.json');arr=np.load(HERE/d/f'{kind}_seed{seed}.npz');p=arr['probabilities'];y=arr['labels']
                np.testing.assert_array_equal(y,val['labels'].numpy());np.testing.assert_array_equal(arr['raw_indices'],val['raw_indices'].numpy())
                np.testing.assert_allclose(p.sum(1),1,atol=1e-6);assert np.isfinite(p).all() and p.min()>=0
                actual=s.evaluate(d,p,val,table)
                for key in s.FIELDS:assert abs(actual[key]-record['metrics'][key])<1e-8
                assert abs(100*f1_score(y,p.argmax(1),average='macro')-record['metrics']['f1'])<1e-8
                cm=confusion_matrix(y,p.argmax(1));assert abs(100*(1-cm[0,0]/cm[0].sum())-record['metrics']['normal_fpr'])<1e-8
                if kind=='with_head':
                    direct=s.rank(arr['node_scores'],val)
                    for key in ['top1','mrr','ndcg5']:assert abs(direct[key]-record['metrics']['direct_node'][key])<1e-8
                else:
                    np.testing.assert_allclose(arr['node_scores'],p@table,atol=2e-7)
                state=torch.load(HERE/d/f'{kind}_seed{seed}.pt',weights_only=True)
                has=any('root_scorer' in k for k in state)
                assert has==(kind=='with_head')==record['has_node_head']
                checked+=1
        diagnostic=u.read(HERE/'full_pipeline_diagnostic.json')['datasets'][d]
        for kind in ['with_head','without_head']:
            for row in diagnostic[kind]['rows']:
                arr=np.load(HERE/d/f"full_pipeline_{kind}_seed{row['seed']}.npz")
                np.testing.assert_array_equal(arr['labels'],val['labels'].numpy())
                np.testing.assert_array_equal(arr['raw_indices'],val['raw_indices'].numpy())
                p=arr['probabilities'];np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
                np.testing.assert_allclose(arr['node_scores'],p@table,atol=2e-7)
                actual=s.evaluate(d,p,val,table)
                for key in s.FIELDS:assert abs(actual[key]-row['metrics'][key])<1e-8
                checked+=1
        details[d]={'validation_rows':len(val['labels']),'eligible_rows':int((val['root_mask']>0).sum()),'lookup_table_fit_rows':len(train['labels'])}
    manifest=u.read(ROOT/'ranking_comparison_20260929/manifest.json')
    unchanged={};intentionally_modified={}
    for path,digest in manifest['sha256'].items():
        if path.startswith('ranking_comparison_20260929'):continue
        if path.startswith('ESWA-round2'):
            intentionally_modified[path]={'before':digest,'after':sha(path)}
            continue
        assert sha(path)==digest,path;unchanged[path]=True
    out={'all_passed':True,'prediction_files_checked':checked,'details':details,'test_accessed':False,'original_model_and_metric_files_unchanged':unchanged,'manuscript_files_intentionally_modified_this_turn':intentionally_modified,'results_sha256':sha(HERE/'results.json'),'full_pipeline_diagnostic_sha256':sha(HERE/'full_pipeline_diagnostic.json'),'source_sha256':sha(HERE/'run.py'),'verifier_sha256':sha(__file__)}
    u.dump(HERE/'verification.json',out);print('VERIFIED',checked,'validation prediction files',flush=True)

if __name__=='__main__':
    torch.set_num_threads(2)
    with u.threadpool_limits(limits=2):main()
