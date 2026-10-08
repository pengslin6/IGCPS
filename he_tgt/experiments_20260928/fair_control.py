"""DA-TGT versus the retained DT-GNN adapter under matched joint optimization."""
from __future__ import annotations
import argparse
import gc
import json
import time
from types import SimpleNamespace
from dataclasses import asdict
from common import *
from scipy import stats
from sklearn.metrics import classification_report
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

MODELS = ["DA-TGT", "DT-GNN"]
LEARNING_RATES = [0.0007, 0.002]
MAX_EPOCHS = 18
OUT = HERE / "fair_control"


class DTGNNJoint(nn.Module):
    def __init__(self, dim, meta):
        super().__init__()
        builder = SimpleNamespace(temporal_feat_dim=dim, n_nodes=meta["n_nodes"], n_phy=meta["n_nodes"]-meta["n_net"])
        self.base = c.exp.build_model("DT-GNN", builder, len(meta["label_names"]))
        self.root_head = c.exp.SharedRootHead(32)
        # These legacy branches are absent from the joint forward/loss.
        for module in (self.base.node_scorer, self.base.gnn_encoder.decoder):
            module.requires_grad_(False)

    def forward(self, features, adjacency):
        if self.training:
            features = features + torch.randn_like(features) * 0.02
        raw_nodes = self.base.encode(features, adjacency)
        nodes = raw_nodes + self.base.feature_enhance(raw_nodes)
        weights = self.base.graph_pool(nodes).squeeze(-1).softmax(dim=1)
        pooled = (weights.unsqueeze(-1) * nodes).sum(dim=1) + 0.3 * nodes.max(dim=1).values
        attended, _ = self.root_head.attention(raw_nodes, raw_nodes, raw_nodes)
        return {"anomaly_logits": self.base.classifier(pooled),
                "root_logits": self.root_head.scorer(attended).squeeze(-1)}


def build_model(name, cache, meta, seed):
    c.joint.set_seed(seed)
    dim = cache["features"].shape[-1]
    if name == "DA-TGT":
        return c.joint.DATGTJoint(dim, meta["n_nodes"], meta["n_net"], len(meta["label_names"]), c.CANDIDATE)
    return DTGNNJoint(dim, meta)


def common_loss(output, labels, targets, eligible, recency, class_weights):
    cls = (F.cross_entropy(output["anomaly_logits"], labels, weight=class_weights, reduction="none") * recency).mean()
    # Use exactly the same objective and eligibility for both methods.
    root = c.supervised_root_loss(output["root_logits"], targets, eligible, labels)
    return cls + 0.30 * root, root


def checks(dataset, cache, meta, adjacency):
    result = {}
    for name in MODELS:
        model = build_model(name, cache, meta, 42)
        model.eval()
        indices = torch.cat([torch.nonzero(cache["labels"] == i, as_tuple=True)[0][:2] for i in range(len(meta["label_names"]))])
        sample = cache["features"][indices]
        output = model(sample, adjacency)
        if name == "DT-GNN":
            with torch.no_grad():
                original_logits = torch.stack([model.base(x, adjacency)["anomaly_logits"] for x in sample])
                original_roots = torch.stack([model.root_head(model.base.encode(x, adjacency)) for x in sample])
            torch.testing.assert_close(output["anomaly_logits"], original_logits, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(output["root_logits"].sigmoid(), original_roots, atol=2e-6, rtol=2e-5)
        _, root = common_loss(output, cache["labels"][indices], cache["root_targets"][indices],
            cache["root_mask"][indices], cache["recency"][indices], c.joint.class_weights(cache["labels"]))
        encoder = model.input_projection if name == "DA-TGT" else model.base.gnn_encoder.input_proj
        grads = torch.autograd.grad(root, list(encoder.parameters()), allow_unused=False)
        norm = sum(float(g.abs().sum()) for g in grads)
        assert norm > 0 and np.isfinite(norm)
        result[name] = {"node_loss_reaches_encoder": True, "encoder_gradient_L1": norm,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "batched_single_graph_equivalence": name == "DT-GNN"}
    # Unknown attack rows must have no gradient from the node loss.
    logits = torch.zeros(3, 4, requires_grad=True)
    labels = torch.tensor([0, 1, 2])
    targets = torch.tensor([[0.,0.,0.,0.], [1.,0.,0.,0.], [0.,0.,0.,0.]])
    loss = c.supervised_root_loss(logits, targets, torch.tensor([0.,1.,0.]), labels)
    grad = torch.autograd.grad(loss, logits)[0]
    assert grad[0].abs().sum() > 0 and grad[1].abs().sum() > 0 and grad[2].abs().sum() == 0
    result["unknown_attack_mask_verified"] = True
    dump(OUT / dataset / "implementation_checks.json", result)
    print(f"IMPLEMENTATION CHECKS PASSED {dataset}", flush=True)


def fit(name, cache, meta, adjacency, seed, lr, epochs, prefix, validation=None):
    start = time.perf_counter()
    model = build_model(name, cache, meta, seed)
    weights = c.joint.class_weights(cache["labels"])
    loader = DataLoader(TensorDataset(*(cache[k] for k in ["features","labels","root_targets","root_mask","recency"])),
        batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=lr * 0.1)
    history, best_key, best_state = [], None, None
    lookup = c.class_root_table(cache, len(meta["label_names"]))
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total = 0.
        for features, labels, targets, eligible, recency in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(features, adjacency)
            loss, _ = common_loss(output, labels, targets, eligible, recency, weights)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach()) * len(labels)
        scheduler.step()
        row = {"epoch":epoch,"loss":total/len(cache["labels"]),"train_seconds":time.perf_counter()-epoch_start}
        if validation is not None:
            probs, roots = c.collect(model, validation, adjacency)
            row.update(c.evaluate(probs, roots, validation, lookup))
            row["selection_score"] = c.balanced_key(row)
            key = (row["selection_score"], row["auc"], -epoch)
            if best_key is None or key > best_key:
                best_key = key
                best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
        history.append(row)
        dump(prefix.with_suffix(".history.json"), history)
        note = f" validation F1/Top1={row['f1']:.2f}/{row['top1']:.2f}" if validation is not None else ""
        print(f"{prefix.name} epoch={epoch}/{epochs} loss={row['loss']:.5f} seconds={row['train_seconds']:.2f}{note}", flush=True)
    if validation is not None:
        model.load_state_dict(best_state)
    torch.save({"model":name,"seed":seed,"learning_rate":lr,"state_dict":model.state_dict(),
        "candidate":asdict(c.CANDIDATE),"elapsed_seconds":time.perf_counter()-start,
        "protocol_sha256":sha(HERE / "PROTOCOL.md")}, prefix.with_suffix(".pt"))
    return model, history, time.perf_counter()-start


def run(dataset):
    out = OUT / dataset
    out.mkdir(parents=True, exist_ok=True)
    meta = load_meta(dataset)
    adjacency = torch.load(HERE / "data_cache" / dataset / "adjacency.pt", weights_only=True)
    train, validation = load_cache(dataset,"train"), load_cache(dataset,"validation")
    checks(dataset, train, meta, adjacency)
    manifest = {"dataset":dataset,"data_sha256":DATA[dataset]["sha256"],"protocol_sha256":sha(HERE/"PROTOCOL.md"),
        "runner_sha256":sha(__file__),"common_sha256":sha(HERE/"common.py"),
        "source_sha256":meta["code_sha256"],"torch":torch.__version__,"threads":torch.get_num_threads(),
        "models":MODELS,"learning_rates":LEARNING_RATES,"validation_epochs":MAX_EPOCHS,
        "selection_seed":42,"final_seeds":SEEDS,"test_status":"retrospective",
        "input_noise":0.02,"calibration":False,"pretraining":False,"frozen_encoder":False,
        "loss":"weighted recency CE + .30*(positive-weighted eligible node BCE + .03*true-normal node BCE)",
        "native_dropout_preserved":True,"identical_architectures_or_parameter_counts":False,
        "run_started":time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if (out/"manifest.json").exists():
        old=json.loads((out/"manifest.json").read_text())
        for k in ("runner_sha256","common_sha256","protocol_sha256","data_sha256"):
            assert old[k]==manifest[k], f"Changed protocol/code while resuming: {k}"
    else:
        dump(out/"manifest.json",manifest)
    lock_path = out/"selection_lock.json"
    if not lock_path.exists():
        selections={}
        for name in MODELS:
            candidates=[]
            for lr in LEARNING_RATES:
                prefix=out/f"validation__{name}__lr{lr}"
                result_path=prefix.with_suffix(".result.json")
                # Decimal points in learning-rate stems require appended suffixes.
                prefix=out/f"validation__{name}__lr{int(lr*1000000)}"
                result_path=prefix.with_suffix(".result.json")
                if result_path.exists():
                    candidates.append(json.loads(result_path.read_text()))
                    continue
                model,history,elapsed=fit(name,train,meta,adjacency,42,lr,MAX_EPOCHS,prefix,validation)
                best=max(history,key=lambda r:(r["selection_score"],r["auc"],-r["epoch"]))
                result={"model":name,"learning_rate":lr,"epoch":best["epoch"],"selection_score":best["selection_score"],
                    "auc":best["auc"],"validation":best,"elapsed_seconds":elapsed}
                dump(result_path,result)
                candidates.append(result)
                del model
                gc.collect()
            selections[name]=max(candidates,key=lambda r:(r["selection_score"],r["auc"],-r["epoch"],-r["learning_rate"]))
        dump(lock_path,{"dataset":dataset,"test_used_for_this_selection":False,"retrospective_dataset":True,
            "selected":selections,"locked_at":time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    lock=json.loads(lock_path.read_text())
    del train,validation
    gc.collect()
    development,test=load_cache(dataset,"development"),load_cache(dataset,"test")
    lookup=c.class_root_table(development,len(meta["label_names"]))
    final_records=[]
    for name in MODELS:
        chosen=lock["selected"][name]
        for seed in SEEDS:
            prefix=out/f"final__{name}__seed{seed}"
            result_path=prefix.with_suffix(".result.json")
            if result_path.exists():
                final_records.append(json.loads(result_path.read_text()))
                continue
            model,history,elapsed=fit(name,development,meta,adjacency,seed,chosen["learning_rate"],chosen["epoch"],prefix)
            probs,roots=c.collect(model,test,adjacency)
            metrics=c.evaluate(probs,roots,test,lookup)
            per_class=classification_report(test["labels"].numpy(),probs.argmax(1),target_names=meta["label_names"],output_dict=True,zero_division=0)
            result={"dataset":dataset,"model":name,"seed":seed,"epochs":chosen["epoch"],"learning_rate":chosen["learning_rate"],
                "training_seconds":elapsed,"selection_lock_sha256":sha(lock_path),"metrics":metrics,"per_class":per_class,
                "checkpoint_sha256":sha(prefix.with_suffix(".pt"))}
            np.savez_compressed(prefix.with_suffix(".predictions.npz"),labels=test["labels"].numpy(),probabilities=probs,root_scores=roots,
                raw_indices=test["raw_indices"].numpy())
            dump(result_path,result)
            final_records.append(result)
            print(f"FINAL {dataset} {name} seed={seed}: "+json.dumps({m:metrics[m] for m in METRICS}),flush=True)
            del model
            gc.collect()
    pd.json_normalize(final_records).to_csv(out/"final_runs.csv",index=False)
    summary={}
    for name in MODELS:
        summary[name]={}
        rows=[r for r in final_records if r["model"]==name]
        assert sorted(r["seed"] for r in rows)==SEEDS
        for metric in METRICS:
            a=np.asarray([r["metrics"][metric] for r in rows])
            summary[name][metric]={"mean":float(a.mean()),"sd":float(a.std(ddof=1))}
    paired={}
    for metric in METRICS:
        diffs=np.asarray([next(r["metrics"][metric] for r in final_records if r["seed"]==s and r["model"]=="DA-TGT")-
                          next(r["metrics"][metric] for r in final_records if r["seed"]==s and r["model"]=="DT-GNN") for s in SEEDS])
        half=float(stats.t.ppf(.975,4)*diffs.std(ddof=1)/np.sqrt(5))
        paired[metric]={"mean":float(diffs.mean()),"sd":float(diffs.std(ddof=1)),
            "ci95_low":float(diffs.mean()-half),"ci95_high":float(diffs.mean()+half),"seed_differences":diffs.tolist()}
    dump(out/"summary.json",{"means":summary,"paired_DA_TGT_minus_DT_GNN":paired,"interpretation":"Retrospective seed variability under matched joint optimization"})
    print(f"FAIR CONTROL COMPLETE {dataset}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",nargs="+",choices=list(DATA),default=list(DATA))
    args=parser.parse_args()
    torch.set_num_threads(2)
    for dataset in args.dataset:
        run(dataset)
