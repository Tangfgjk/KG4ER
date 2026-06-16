from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

TARGETS = [
    ROOT / "ER" / "KG4ER" / "data" / "step1_cal_recommend.py",
    ROOT / "ER" / "KG4ER" / "data" / "step2_createtriples.py",
    ROOT / "ER" / "KG4ER" / "data" / "step3_get_kc_response.py",
    ROOT / "ER" / "KG4ER" / "codes" / "run.py",
    ROOT / "ER" / "KG4ER" / "codes" / "test_TransE.py",
]

OPTIONAL_TARGETS = [
    ROOT / "ER" / "pykt-toolkit-main" / "data" / "Eedi" / "calculate_time.py",
]


def iter_existing_targets():
    for path in TARGETS + OPTIONAL_TARGETS:
        if path.exists():
            yield path


def test_er_eedi_scripts_do_not_use_local_absolute_windows_paths():
    for path in iter_existing_targets():
        text = path.read_text(encoding="utf-8")
        assert "E:\\" not in text, f"{path} still contains a local absolute Windows path"


def test_er_eedi_scripts_default_to_canonical_file_names():
    canonical_files = {
        "stu2ex_forget.json",
        "stu2ex_recommend.json",
        "triples.txt",
        "test_triples.txt",
    }

    combined = "\n".join(path.read_text(encoding="utf-8") for path in iter_existing_targets())
    for name in canonical_files:
        assert name in combined

    legacy_defaults = {
        "stu2ex_forget_ori.json",
        "stu2ex_recommend_ori.json",
        "triples_new_ori.txt",
        "test_triples_new_ori.txt",
    }
    for name in legacy_defaults:
        assert name not in combined, f"{name} should not be a default ER/Eedi path"
