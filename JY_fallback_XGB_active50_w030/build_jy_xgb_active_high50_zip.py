"""Build a standalone JY + frozen fallback-XGB candidate without altering champion."""
from pathlib import Path
import shutil, tempfile, zipfile

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'parent'/'submit_jy_runners_high_li_bridge027_public1172.zip'
ASSET=ROOT/'frozen_asset'
RUNTIME=ROOT/'scripts'/'fallback_xgb_frozen_runtime.py'
OUT=ROOT/'rebuilt'/'submit_jy_fallback_xgb_active50_w030.zip'

def main():
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='jy_xgb_') as td:
        stage=Path(td)
        with zipfile.ZipFile(SOURCE) as z: z.extractall(stage)
        script=stage/'script.py'; text=script.read_text(encoding='utf-8')
        text=text.replace('BRIDGE_SCALE = 1.2\n','BRIDGE_SCALE = 1.2\nXGB_ACTIVE_WEIGHT = 0.30\nXGB_ACTIVE_THRESHOLD = 0.50\n')
        marker='\ndef _window_adjustment(\n'
        helper='''\ndef _predict_fallback_xgb(frame: pd.DataFrame) -> np.ndarray:\n    module = _load_module("jy_fallback_xgb", MODEL_DIR / "fallback_xgb" / "runtime.py")\n    return np.asarray(module.predict(frame, MODEL_DIR / "fallback_xgb"), dtype=np.float64)\n\n'''
        text=text.replace(marker,helper+marker)
        old='''    output[active] = np.clip(\n        (1.0 - H1_WEIGHT) * effective_bridge[active]\n        + H1_WEIGHT * h1[active]\n        + C3_WEIGHT * c3[active],\n        0.001,\n        0.999,\n    )\n    return parent, h1, c3, active, output\n'''
        new='''    output[active] = np.clip(\n        (1.0 - H1_WEIGHT) * effective_bridge[active]\n        + H1_WEIGHT * h1[active]\n        + C3_WEIGHT * c3[active],\n        0.001,\n        0.999,\n    )\n    xgb_active = active & (output >= XGB_ACTIVE_THRESHOLD)\n    if xgb_active.any():\n        xgb_probability = _predict_fallback_xgb(frame)\n        output[xgb_active] = np.clip(\n            (1.0 - XGB_ACTIVE_WEIGHT) * output[xgb_active]\n            + XGB_ACTIVE_WEIGHT * xgb_probability[xgb_active], 0.001, 0.999\n        )\n    return parent, h1, c3, active, output\n'''
        if old not in text: raise RuntimeError('champion formula marker missing')
        script.write_text(text.replace(old,new),encoding='utf-8')
        target=stage/'model'/'fallback_xgb'; target.mkdir(parents=True,exist_ok=True)
        for name in ('fallback_xgb.json','feature_columns.json','fallback_lookups.joblib','metadata.json'):
            shutil.copy2(ASSET/name,target/name)
        shutil.copy2(RUNTIME,target/'runtime.py')
        req=stage/'requirements.txt'; req.write_text(req.read_text(encoding='utf-8').rstrip()+'\nxgboost==3.2.0\n',encoding='utf-8')
        with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED) as z:
            for p in stage.rglob('*'):
                if p.is_file(): z.write(p,p.relative_to(stage).as_posix())
    print(OUT)

if __name__=='__main__': main()
