"""Measure training emissions of the final configuration (one seed per family, both anchor sets) with CodeCarbon."""
import subprocess, json, os, sys
from codecarbon import EmissionsTracker
os.makedirs("report_final",exist_ok=True); res={}
for FS in ("allnoll","v5x_noll"):
    for M,R in (("lgb","420"),("xgb","360"),("cat","975"),("mlp","")):
        tr=EmissionsTracker(project_name=f"{M}_{FS}",output_dir="report_final",log_level="error"); tr.start()
        env={**os.environ,"SEED":"0","TAG":"_carbon"}; 
        if M=="mlp": env["EPOCHS"]="1"
        subprocess.run(["./.venv/bin/python","run_models.py","FINAL",M,FS]+([R] if R else []),env=env,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        res[f"{M}_{FS}"]=tr.stop()
json.dump({"kg_co2e_per_model":res,"total_one_seed_each":sum(res.values()),"note":"final ensemble = 3 lgb + 3 xgb + 2 cat + 4 mlp seeds per anchor set"},open("report_final/carbon.json","w"),indent=1); print(res)
