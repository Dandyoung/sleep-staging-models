#!/usr/bin/env python3
# (shebang, interpreter 설정)

'''
- verify_mesa_dataset.py
MESA 데이터셋에서 XML(annotations)과 EDF(신호) 파일의 ID 매칭을 검증하고 결과를 JSON으로 저장하는 모듈

- 사용 예시 1)
python verify_mesa_dataset.py \
    --xml-dir ./mesa-commercial-use/polysomnography/annotations-events-nsrr \
    --edf-dir ./mesa-commercial-use/polysomnography/edfs \
    --out ./paircheck_result.json

- 사용 예시 2)
chmod +x verify_mesa_dataset.py
python verify_mesa_dataset.py \
    --xml-dir ./mesa-commercial-use/polysomnography/annotations-events-nsrr \
    --edf-dir ./mesa-commercial-use/polysomnography/edfs \
    --out ./paircheck_result.json
'''
    
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ID_PATTERN = re.compile(r"mesa[-_]?sleep[-_]?(\d+)", re.IGNORECASE)

def extract_id(filename: str) -> str | None:
    """파일명에서 mesa-sleep-#### 숫자 ID 추출 (선행 0 유지)."""
    m = ID_PATTERN.search(filename)
    return m.group(1) if m else None

def index_by_id(dir_path: Path, allow_exts: Tuple[str, ...]) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:
    """
    디렉토리 내 허용 확장자의 파일을 훑어 ID -> 파일경로 매핑 생성.
    같은 ID가 여러 파일에 존재하면 duplicates에 기록.
    """
    id_to_file: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = defaultdict(list)

    for p in dir_path.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in allow_exts:
            continue
        rec_id = extract_id(p.name)
        if not rec_id:
            continue
        if rec_id in id_to_file:
            if not duplicates[rec_id]:
                duplicates[rec_id] = [id_to_file[rec_id]]
            duplicates[rec_id].append(p)
        else:
            id_to_file[rec_id] = p

    return id_to_file, duplicates

def to_filenames(paths: List[Path]) -> List[str]:
    """Path 리스트를 파일명 리스트로 변환."""
    return [p.name for p in paths]

def compare_pairs(xml_dir: Path, edf_dir: Path) -> dict:
    xml_map, xml_dups = index_by_id(xml_dir, allow_exts=(".xml",))
    edf_map, edf_dups = index_by_id(edf_dir, allow_exts=(".edf",))

    xml_ids = set(xml_map.keys())
    edf_ids = set(edf_map.keys())

    matched_ids   = sorted(xml_ids & edf_ids, key=lambda x: (len(x), x))
    only_xml_ids  = sorted(xml_ids - edf_ids, key=lambda x: (len(x), x))
    only_edf_ids  = sorted(edf_ids - xml_ids, key=lambda x: (len(x), x))

    result = {
        "summary": {
            "xml_dir": str(xml_dir),
            "edf_dir": str(edf_dir),
            "xml_count": len(xml_map),
            "edf_count": len(edf_map),
            "matched": len(matched_ids),
            "only_xml": len(only_xml_ids),
            "only_edf": len(only_edf_ids),
            # 중복은 "ID: 갯수"로 요약
            "xml_duplicates": {k: len(v) for k, v in xml_dups.items()},
            "edf_duplicates": {k: len(v) for k, v in edf_dups.items()},
        },
        # 본문에는 파일 "경로" 대신 "파일명"만 저장
        "matched": [
            {"id": rid, "xml": xml_map[rid].name, "edf": edf_map[rid].name}
            for rid in matched_ids
        ],
        "only_xml": [
            {"id": rid, "xml": xml_map[rid].name}
            for rid in only_xml_ids
        ],
        "only_edf": [
            {"id": rid, "edf": edf_map[rid].name}
            for rid in only_edf_ids
        ],
        # 참고용: 중복의 상세 파일명 리스트(필요 없으면 이 블록 삭제해도 됨)
        "duplicates": {
            "xml": {rid: to_filenames(paths) for rid, paths in xml_dups.items()},
            "edf": {rid: to_filenames(paths) for rid, paths in edf_dups.items()},
        },
    }
    return result

def main():
    ap = argparse.ArgumentParser(description="Verify MESA dataset (XML<->EDF pairing) and export JSON (filenames only).")
    ap.add_argument("--xml-dir", required=True, help="annotations-events-nsrr 디렉토리 경로")
    ap.add_argument("--edf-dir", required=True, help="edfs 디렉토리 경로")
    ap.add_argument("--out", required=True, help="결과 JSON 경로")
    args = ap.parse_args()

    xml_dir = Path(args.xml_dir).resolve()
    edf_dir = Path(args.edf_dir).resolve()

    if not xml_dir.is_dir():
        raise SystemExit(f"[ERROR] XML 디렉토리 없음: {xml_dir}")
    if not edf_dir.is_dir():
        raise SystemExit(f"[ERROR] EDF 디렉토리 없음: {edf_dir}")

    result = compare_pairs(xml_dir, edf_dir)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
