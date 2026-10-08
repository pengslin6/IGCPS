import copy
import random
import warnings
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
GROUPS={"IGCPS":[0,2],"TE-CUP-SEC":[0,6]}
EPOCHS=[30,60,120]

def features(dataset, setting, arr):
    key = "history" if setting["kind"].startswith("history") else "static"
    x = arr[key]
    if setting["view"] == "network":
        net, width = (57 if dataset == "IGCPS" else 21), arr["static"].shape[1]
        ids = np.arange(net) if key == "static" else np.r_[np.concatenate([np.arange(net)+block*width for block in range(4)]), np.arange(width*4,width*4+3)]
        x = x[:, ids]
    return np.asarray(x, dtype=np.float64)


def network(width):
    return torch.nn.Sequential(torch.nn.Linear(width,128),torch.nn.GELU(),torch.nn.Dropout(.15),
        torch.nn.Linear(128,64),torch.nn.GELU(),torch.nn.Dropout(.15),torch.nn.Linear(64,2))


def fit(dataset, setting, arr, seed):
    group = GROUPS[dataset]
    mask = np.isin(arr["labels"], group)
    x = features(dataset, setting, arr)[mask]
    scaler = StandardScaler().fit(x)
    values = np.clip(scaler.transform(x), -10, 10)
    labels = arr["labels"][mask]
    recency = arr["recency"][mask].astype(np.float64)
    head = {"dataset": dataset, "setting": setting, "scaler": scaler, "seed": seed,
        "fitting_counts": {str(k): int((labels == k).sum()) for k in group}}
    if setting["kind"] == "history_lr":
        model = LogisticRegression(C=1,solver="lbfgs",class_weight="balanced",tol=1e-8,max_iter=3000,random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(values, labels, sample_weight=recency)
        np.testing.assert_array_equal(model.classes_, group)
        yield 0, {**head, "model": model}
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = network(values.shape[1])
    x_tensor = torch.tensor(values,dtype=torch.float32)
    y_tensor = torch.tensor(np.searchsorted(group,labels),dtype=torch.long)
    weights = torch.tensor([len(labels)/(2*(labels == k).sum()) for k in group],dtype=torch.float32)
    sample_weights = weights[y_tensor]*torch.tensor(recency,dtype=torch.float32)
    sample_weights /= sample_weights.mean()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=120)
    for epoch in range(1,121):
        model.train()
        order = torch.randperm(len(labels))
        losses = []
        for ids in order.split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = (torch.nn.functional.cross_entropy(model(x_tensor[ids]),y_tensor[ids],reduction="none")*sample_weights[ids]).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        if epoch in EPOCHS:
            yield epoch, {**head,"setting":{**setting,"epoch":epoch,"name":f"{setting['view']}__{setting['kind']}__e{epoch:03d}"},
                "state_dict":copy.deepcopy(model.state_dict()),"last_train_loss":float(np.mean(losses))}


def predict(head, arr):
    x = np.clip(head["scaler"].transform(features(head["dataset"],head["setting"],arr)),-10,10)
    if "model" in head:
        return head["model"].predict_proba(x)
    # Evaluation must not consume the random stream used by the continuing fit.
    with torch.random.fork_rng(devices=[]):
        model = network(x.shape[1])
    model.load_state_dict(head["state_dict"])
    model.eval()
    with torch.no_grad():
        return torch.cat([model(batch).softmax(-1) for batch in torch.tensor(x,dtype=torch.float32).split(1024)]).numpy()


