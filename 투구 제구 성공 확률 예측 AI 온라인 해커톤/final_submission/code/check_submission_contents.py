"""Read-only submission completeness check. This is not organizer approval."""
import argparse
import csv
import json
from pathlib import Path
import zipfile

COMPONENT_NAMES = {
    "v124 and v124_bridge parent ensembles", "H1 three-seed CatBoost",
    "C3 sign agreement", "fallback XGBoost", "recent futures five-seed CatBoost",
    "futures and anchor lowrank lookup", "player transition lookup",
    "workload H1", "Beta cell and final assembly",
}


def check(package, pptx=None, attendance=None, code_only=False):
    required = {}
    diagnostics = {}
    required["environment_document"] = (package / "01_ENVIRONMENT.md").is_file()
    required["training_entrypoint"] = (package / "code/train.py").is_file()
    required["reproduction_instructions"] = (package / "03_REPRODUCE.md").is_file()
    for name, filename, key in (
        ("h1_fresh_fit", "h1_fresh_fit.json", "strict_fit_parity"),
        ("futures_fresh_fit", "futures_fresh_fit.json", "strict_fit_parity"),
        ("xgb_fresh_fit", "xgb_fresh_fit.json", "exact_fit_parity"),
    ):
        path = package / "verification" / filename
        diagnostics[name] = path.is_file() and json.loads(path.read_text(encoding="utf-8")).get(key) is True
    coverage = package / "verification/training_coverage.json"
    coverage_data = json.loads(coverage.read_text(encoding="utf-8")) if coverage.is_file() else {}
    components = coverage_data.get("components", [])
    diagnostics["all_components_exact_fresh_fit_parity"] = (
        len(components) == len(COMPONENT_NAMES)
        and {c.get("name") for c in components} == COMPONENT_NAMES
        and all(
        c.get("fresh_training_verified") is True for c in components))
    required["solution_pptx_structure"] = False
    if pptx and pptx.is_file() and pptx.suffix.lower() == ".pptx":
        try:
            with zipfile.ZipFile(pptx) as archive:
                names = archive.namelist()
                required["solution_pptx_structure"] = (
                    archive.testzip() is None and "ppt/presentation.xml" in names
                    and any(n.startswith("ppt/slides/slide") and n.endswith(".xml") for n in names))
        except zipfile.BadZipFile:
            pass
    required["attendance_entries"] = False
    if attendance and attendance.is_file():
        with attendance.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        required["attendance_entries"] = (
            1 <= len(rows) <= 5 and len({r.get("member", "").strip() for r in rows}) == len(rows)
            and all(r.get("member", "").strip() and r.get("phase3_attendance") in
                    ("attending", "not_attending") for r in rows))
    separately_provided = []
    if code_only:
        separately_provided = ["solution_pptx_structure", "attendance_entries"]
        for name in separately_provided:
            required.pop(name)
    return {
        "scope": "code_only" if code_only else "complete_submission",
        "separately_provided": separately_provided,
        "required_checks": required,
        "diagnostic_checks": diagnostics,
        "missing_required": [name for name, ok in required.items() if not ok],
        "unverified_diagnostics": [name for name, ok in diagnostics.items() if not ok],
        "local_requirements_passed": all(required.values()), "organizer_approval": False,
        "notes":[
            "Model parity checks use the existing recorded reports; no training is executed here.",
            "PPT structure and attendance completeness do not validate slide accuracy, identities, or attendance eligibility.",
            "Exact component parity is reported separately because it is a conservative local diagnostic, not a published organizer requirement.",
            "Historical submission compliance and official Private Score require separate review."
        ]
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--pptx",type=Path)
    parser.add_argument("--attendance",type=Path)
    parser.add_argument("--code-only", action="store_true",
                        help="Check code only; PPT and attendance will be provided separately.")
    args = parser.parse_args()
    report = check(args.package_dir,args.pptx,args.attendance,args.code_only)
    print(json.dumps(report,indent=2))
    if not report["local_requirements_passed"]:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
