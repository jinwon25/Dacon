import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load():
    spec = importlib.util.spec_from_file_location("contents_check",ROOT / "check_submission_contents.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_empty_package_is_not_complete(tmp_path):
    report = load().check(tmp_path)
    assert not report["local_requirements_passed"]
    assert not report["organizer_approval"]
    assert "all_components_exact_fresh_fit_parity" in report["unverified_diagnostics"]

def test_one_component_report_cannot_stand_for_all_training(tmp_path):
    verification = tmp_path / "verification"
    verification.mkdir()
    (verification / "h1_fresh_fit.json").write_text(json.dumps({"strict_fit_parity":True}),encoding="utf-8")
    report = load().check(tmp_path)
    assert report["diagnostic_checks"]["h1_fresh_fit"]
    assert not report["diagnostic_checks"]["all_components_exact_fresh_fit_parity"]

def test_blank_or_unknown_attendance_is_rejected(tmp_path):
    csv_file = tmp_path / "attendance.csv"
    csv_file.write_text("member,phase3_attendance\nmember1,unknown\n",encoding="utf-8")
    assert not load().check(tmp_path,attendance=csv_file)["required_checks"]["attendance_entries"]
    csv_file.write_text("member,phase3_attendance\nmember1,attending\n",encoding="utf-8")
    assert load().check(tmp_path,attendance=csv_file)["required_checks"]["attendance_entries"]


def test_code_only_excludes_external_attachments_but_keeps_training_checks(tmp_path):
    report = load().check(tmp_path, code_only=True)
    assert report["scope"] == "code_only"
    assert set(report["separately_provided"]) == {"solution_pptx_structure", "attendance_entries"}
    assert "solution_pptx_structure" not in report["required_checks"]
    assert "attendance_entries" not in report["required_checks"]
    assert "all_components_exact_fresh_fit_parity" in report["unverified_diagnostics"]
    assert not report["local_requirements_passed"]


def test_partial_coverage_cannot_be_marked_complete(tmp_path):
    verification = tmp_path / "verification"
    verification.mkdir()
    (verification / "training_coverage.json").write_text(json.dumps({
        "components": [{"name": "H1 three-seed CatBoost", "fresh_training_verified": True}]
    }), encoding="utf-8")
    assert not load().check(tmp_path, code_only=True)["diagnostic_checks"]["all_components_exact_fresh_fit_parity"]


def test_integrated_package_is_code_ready_but_keeps_diagnostics_separate():
    report = load().check(ROOT.parent, code_only=True)
    assert report["local_requirements_passed"]
    assert not report["diagnostic_checks"]["xgb_fresh_fit"]
    assert "all_components_exact_fresh_fit_parity" in report["unverified_diagnostics"]
    assert not report["organizer_approval"]