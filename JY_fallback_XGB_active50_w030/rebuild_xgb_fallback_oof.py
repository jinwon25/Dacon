"""Build XGB OOF with the exact same fallback TrackMan feature path used for test.

Unlike the cached X98 OOF, this deliberately disables the train-row exact TM5
cache.  Thus the OOF feature contract matches a future test-season inference
path: prior-season official history plus a row-local pitcher lookup.
"""
from pathlib import Path
import sys,numpy as np,pandas as pd,xgboost as xgb
ROOT=Path(__file__).resolve().parents[1];HY=ROOT.parents[1]/'reference-hyunku-lga-data'/'lga_data-main';sys.path.insert(0,str(HY));import codex_reproduction.repro_v10a_core as rr
OUT=ROOT/'artifacts'/'private_oof_runtime';CACHE=OUT/'hyunku_fallback_profile';CACHE.mkdir(exist_ok=True)
def bss(y,p):r=y.mean();return 1e5*(1-np.mean((y-p)**2)/(r*(1-r)))
def main():
 raw=pd.read_csv(HY/'data'/'train.csv',encoding='utf-8-sig');old=rr.ART;rr.ART=CACHE
 try: base=rr.add_public_context_features(raw,raw,rr.make_base_features(raw,raw))
 finally: rr.ART=old
 y=raw.control_success.to_numpy(np.float32);se=raw.season.to_numpy();isf=raw.game_type.eq('F').to_numpy();params=dict(n_estimators=1800,learning_rate=.006,max_depth=10,min_child_weight=6000,subsample=.7,colsample_bytree=.5,reg_lambda=50.,reg_alpha=1.,tree_method='hist',device='cuda:0',eval_metric='logloss',verbosity=0)
 for year in (2022,2023,2024):
  tr=(se<year)&~(isf&(se<=2022));va=(se==year)&~isf;w=(.5**((year-1-se[tr])/2.)).astype(np.float32);m=xgb.XGBClassifier(**params,random_state=2028);m.fit(base.loc[tr],y[tr],sample_weight=w);p=m.predict_proba(base.loc[va])[:,1];np.save(OUT/f'hyunku_fallback_xgb_{year}.npy',p.astype(np.float32));print(year,'BSS',bss(y[va],p),'rows',len(p),flush=True)
if __name__=='__main__':main()
