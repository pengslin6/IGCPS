"""Check copied provenance and exact decisions against accepted predictions."""
import argparse
import hashlib
import json
import sys
import current_model as c
from current_model import HERE,MODEL,np,torch,read


def main(phase):
    manifest=read(HERE/'migration_manifest.json')
    for record in manifest['artifacts']:
        path=MODEL/record['target']
        with path.open('rb') as f:
            assert hashlib.file_digest(f,'sha256').hexdigest()==record['sha256']
    checks=[]
    for dataset in ['IGCPS','TE-CUP-SEC']:
        cache=torch.load(MODEL.parent/f'experiments_20260928/data_cache/{dataset}/test.pt',weights_only=True)
        with np.load(MODEL/dataset/'test.npz') as arrays:
            history={k:arrays[k] for k in ['static','history']}
        for seed in c.SEEDS:
            out=c.predict(dataset,seed,cache['features'],history)
            with np.load(MODEL/dataset/f'seed{seed}.predictions.npz') as expected:
                for key in ['predictions','incumbent_predictions']:
                    np.testing.assert_array_equal(out[key],expected[key])
                for key in ['root_scores','base_probabilities','conditional_probabilities']:
                    np.testing.assert_allclose(out[key],expected[key],atol=2e-6,rtol=2e-5)
            checks.append({'dataset':dataset,'seed':seed,'rows':len(out['predictions']),'identical_decisions':True})
            print('RETAINED VERIFIED',phase,dataset,seed,flush=True)
    forbidden=['detection_improvement','detection_context','detection_cost_role','detection_hierarchical','detection_history_20260929','detection_transfer','detection_selective']
    imported=[str(getattr(module,'__file__','')) for module in list(sys.modules.values())]
    assert not any(name in path for name in forbidden for path in imported)
    result={'all_passed':True,'phase':phase,'checks':checks,'no_old_experiment_imports':True}
    (HERE/f'verification_{phase}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['before_cleanup','after_cleanup'])
    main(parser.parse_args().phase)
