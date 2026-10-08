"""Source-grounded HE-TGT architecture and representative diagnostics."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=ROOT/'ESWA-round2'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none',
    'axes.spines.top':False,'axes.spines.right':False})


def save(fig,name):
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def architecture():
    # Claim: past-only history refines class probabilities; node ranking is a separate graph output.
    fig,ax=plt.subplots(figsize=(10.3,5.5)); ax.set(xlim=(0,10.3),ylim=(0,5.5)); ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.08',facecolor=color,edgecolor='#516273',lw=.8))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)
    def arrow(points):
        for a,b in zip(points[:-2],points[1:-1]):
            ax.plot([a[0],b[0]],[a[1],b[1]],color='#435466',lw=1)
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=10,color='#435466',lw=1))
    box(.15,2.7,1.55,1.25,'Available traffic\nand process records\n\nBackward-time fusion','#eef1f5')
    box(2.15,3.5,1.75,1.1,'Current typed graph\nEndpoint / process nodes\nK = 1 graph','#dbe8f3')
    box(4.35,3.5,2.15,1.1,'Type adapters\n2 local graph blocks\n2 global Transformer layers','#dbe8f3')
    box(7.0,4.3,1.45,.7,'Node head\nReference scores','#dbe8f3')
    box(8.9,4.3,1.2,.7,'Investigation\nranking','#eef1f5')
    box(7.0,3.15,1.45,.75,'Class head\nGraph probability pG','#dbe8f3')
    box(2.15,.55,1.75,1.6,'Current + past features\nPast: same directed pair\n< current time\n≤ 30 s; ≤ 8 records\nChanges, spread, age','#e7eee3')
    box(4.35,1.7,2.15,.85,'Multiclass history trees\n7 leaves; probability pH','#e7eee3')
    box(4.35,.35,2.15,.85,'Universal class-pair trees\n15 leaves; all class pairs','#e7eee3')
    box(7.0,1.7,1.45,.85,'Probability blend\n0.75 pG + 0.25 pH','#f3e8d6')
    box(8.9,1.7,1.2,.85,'Top-two\nrefinement\nλ = 0.5','#f3e8d6')
    box(8.9,.35,1.2,.75,'Final class\nargmax','#eef1f5')
    arrow([(1.7,3.6),(1.93,3.6),(1.93,4.05),(2.15,4.05)])
    arrow([(1.7,3),(1.93,3),(1.93,1.35),(2.15,1.35)])
    arrow([(3.9,4.05),(4.35,4.05)])
    arrow([(6.5,4.2),(6.75,4.2),(6.75,4.65),(7.,4.65)])
    arrow([(8.45,4.65),(8.9,4.65)])
    arrow([(6.5,3.8),(6.75,3.8),(6.75,3.525),(7.,3.525)])
    arrow([(3.9,1.9),(4.12,1.9),(4.12,2.125),(4.35,2.125)])
    arrow([(3.9,.8),(4.35,.775)])
    arrow([(6.5,2.125),(7.,2.125)])
    arrow([(7.725,3.15),(7.725,2.55)])
    arrow([(8.45,2.125),(8.9,2.125)])
    arrow([(6.5,.775),(8.67,.775),(8.67,1.92),(8.9,1.92)])
    arrow([(9.5,1.7),(9.5,1.1)])
    ax.text(.15,5.25,'HE-TGT: current graph + past-only history',fontsize=12,fontweight='bold')
    ax.text(.15,.02,'Identical history and refinement rules on both datasets; fitted parameters and input dimensions differ.',fontsize=8)
    save(fig,'he_tgt_architecture')


def diagnostics():
    metrics=json.loads((HERE/'metrics.json').read_text(encoding='utf-8'))
    data=np.load(HERE/'IGCPS'/'representative.npz')
    p=data['probabilities']; y=data['labels']; names=metrics['datasets']['IGCPS']['metadata']['label_names']
    confusion=np.zeros((len(names),len(names)),int); np.add.at(confusion,(y,p.argmax(1)),1)
    fig,ax=plt.subplots(figsize=(4.7,3.7)); ax.imshow(confusion,cmap='Blues')
    ax.set(xticks=range(len(names)),yticks=range(len(names)),xticklabels=names,yticklabels=names,xlabel='Predicted class',ylabel='True class')
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j,i,str(confusion[i,j]),ha='center',va='center',color='white' if confusion[i,j]>confusion.max()*.55 else '#172634')
    save(fig,'he_tgt_confusion')
    bins=metrics['datasets']['IGCPS']['rows'][0]['calibration']['bins']
    fig,ax=plt.subplots(figsize=(4.5,3.7));ax.plot([0,1],[0,1],':',color='#64748b')
    ax.scatter([b['confidence'] for b in bins],[b['accuracy'] for b in bins],s=[15+150*b['n']/len(y) for b in bins],color='#356c93')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean confidence',ylabel='Observed accuracy')
    save(fig,'he_tgt_reliability')
    node_names=metrics['datasets']['IGCPS']['metadata']['node_names']
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.9))
    axes=axes.ravel()
    axes[-1].axis('off')
    axes[-1].text(.05,.75,'Blue: acceptable reference\nGray: other node\n\nMP: medium pressure\nHP: high pressure\n\nFirst test window per class\nFixed graph seed 11',fontsize=8,va='top')
    short_names={
        'Phy_Switch status of sub-high pressure switch':'Sub-HP switch',
        'Phy_Switch status of high pressure switch':'HP switch',
        'Phy_Switch status of medium pressure skid heater':'MP heater switch',
        'Phy_Outlet pressure of medium pressure skid sensor':'MP outlet pressure',
        'Phy_Outlet temperature of medium pressure skid heater':'MP heater outlet temp.',
        'Phy_Outlet temperature of medium pressure skid sensor':'MP sensor outlet temp.',
        'Phy_Inlet temperature of medium pressure skid heater':'MP heater inlet temp.',
        'Phy_Outlet pressure of high pressure sensor':'HP outlet pressure',
        'Phy_Flow rate of medium pressure skid flow sensor':'MP flow rate',
    }
    examples=[]
    for label,ax in zip(range(1,6),axes):
        idx=int(np.flatnonzero(y==label)[0]); ranks=np.argsort(data['root_scores'][idx])[::-1][:5]
        labels=[short_names.get(node_names[k],node_names[k].replace('IP_','')) for k in ranks]
        import textwrap
        labels=['\n'.join(textwrap.wrap(n,22)) for n in labels]
        values=data['root_scores'][idx,ranks]; truth=data['root_targets'][idx,ranks]>0
        ax.barh(range(5),values,color=['#417b9e' if t else '#bfc7ce' for t in truth])
        ax.set(yticks=range(5),yticklabels=[f'{k+1}' for k in range(5)],xlim=(0,1.05),xlabel='Node score')
        ax.invert_yaxis(); ax.set_title(f'{names[label]} → {names[p[idx].argmax()]}',fontsize=9)
        for k,(v,label_text) in enumerate(zip(values,labels)):
            ax.text(.015,k,label_text,fontsize=7,va='center',color='#101820')
        examples.append({'class':names[label],'prediction':names[p[idx].argmax()],'raw_index':int(data['raw_indices'][idx]),
            'nodes':[node_names[k] for k in ranks],'scores':values.tolist(),'reference':truth.tolist()})
    fig.suptitle('HE-TGT reference-node scores: top five candidates',fontsize=9,y=1.01)
    fig.tight_layout();save(fig,'he_tgt_node_examples')
    (HERE/'node_examples.json').write_text(json.dumps(examples,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.6))
    for d,ax in zip(['IGCPS','TE-CUP-SEC'],axes):
        history=json.loads((ROOT/'experiments_20260928'/'fair_control'/d/'final__DA-TGT__seed11.history.json').read_text())
        ax.plot([h['epoch'] for h in history],[h['loss'] for h in history],color='#356c93',marker='o',ms=2.5)
        ax.set(title=d,xlabel='Graph-backbone epoch',ylabel='Joint training loss')
    fig.tight_layout();save(fig,'he_tgt_backbone_convergence')


if __name__=='__main__':
    architecture();diagnostics()
