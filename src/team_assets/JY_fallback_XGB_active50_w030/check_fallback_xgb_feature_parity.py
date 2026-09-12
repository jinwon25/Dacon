from pathlib import Path
import sys, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; HY=ROOT.parents[1]/'reference-hyunku-lga-data'/'lga_data-main'; ASSET=ROOT/'artifacts'/'private_oof_runtime'/'jy_fallback_xgb_asset'
sys.path.insert(0,str(HY)); import codex_reproduction.repro_v10a_core as rr
sys.path.insert(0,str(ROOT/'scripts')); import fallback_xgb_frozen_runtime as frozen
def main():
 train=pd.read_csv(HY/'data'/'train.csv',encoding='utf-8-sig'); test=pd.read_csv(HY/'data'/'test.csv',encoding='utf-8-sig')
 old=rr.ART; rr.ART=ASSET
 try: ref=rr.add_public_context_features(test.copy(),train,rr.make_base_features(test.copy(),train))
 finally: rr.ART=old
 ref=ref.reindex(columns=frozen.json.loads((ASSET/'feature_columns.json').read_text(encoding='utf-8')))
 got=frozen.build(test,ASSET)
 d=np.abs(ref.to_numpy(float)-got.to_numpy(float))
 mismatch=int(np.sum(np.isfinite(ref.to_numpy(float)) != np.isfinite(got.to_numpy(float))))
 print({'shape':got.shape,'finite_mismatch':mismatch,'max_abs':float(np.nanmax(d))})
 worst=np.nanmax(d,axis=0).argsort()[-12:][::-1]
 print([(got.columns[i],float(np.nanmax(d[:,i]))) for i in worst])
 print(pd.DataFrame({'ref':ref['p_mid_ssn'],'got':got['p_mid_ssn'],'n':test['asof_pitcher_n'],'rate':test['asof_pitcher_middle_rate'],'pid':test['pitcher_id']}))
 if not np.allclose(ref.to_numpy(float),got.to_numpy(float),equal_nan=True,atol=1e-6,rtol=0): raise SystemExit('feature parity failed')
 print('feature parity passed')
if __name__=='__main__':main()
