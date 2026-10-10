"""Current no-head architecture; preserves the evaluated initialization order."""
from pathlib import Path
from dataclasses import replace
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'detection_history_final_20260929/retained_runtime'))
from backbone import DATGTJoint,CANDIDATE

class NodeFreeGraph(DATGTJoint):
    def __init__(self,dim,meta):
        super().__init__(dim,meta['n_nodes'],meta['n_net'],len(meta['label_names']),
                         replace(CANDIDATE,decouple_detection_root=True,root_weight=0.,normal_root_weight=0.))
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
