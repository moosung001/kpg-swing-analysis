from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, Optional

# 프로젝트 루트 탐색 마커(루트에 존재해야 함)
ROOT_MARKERS = (".project_root", "pyproject.toml")


def _iter_parents(p: Path) -> Iterable[Path]:
    p = p.resolve()
    yield p
    for parent in p.parents:
        yield parent


def find_project_root(start: Optional[Path] = None) -> Path:
    """
    우선순위:
    1) 환경변수 KPG_SWING_ROOT
    2) start(기본: 이 파일 위치)에서 상위로 ROOT_MARKERS 탐색
    3) cwd에서 상위 탐색
    """
    env_root = os.environ.get("KPG_SWING_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"KPG_SWING_ROOT 경로가 존재하지 않습니다: {root}")
        return root

    start = (start or Path(__file__).resolve()).resolve()
    for cand in _iter_parents(start):
        for mk in ROOT_MARKERS:
            if (cand / mk).exists():
                return cand

    cwd = Path.cwd().resolve()
    for cand in _iter_parents(cwd):
        for mk in ROOT_MARKERS:
            if (cand / mk).exists():
                return cand

    raise FileNotFoundError(
        "프로젝트 루트를 찾지 못했습니다. "
        "루트에 .project_root(빈 파일) 또는 pyproject.toml이 있는지 확인하세요. "
        "또는 환경변수 KPG_SWING_ROOT를 지정하세요."
    )


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def case_dir(self) -> Path:
        return self.root / "KPG193_ver1_2"

    @property
    def case_mfile(self) -> Path:
        # 프로젝트에 이미 포함된 MATPOWER m 파일
        return self.case_dir / "network" / "m" / "KPG193_ver1_2.m"

    @property
    def static_dir(self) -> Path:
        return self.root / "data_static"

    @property
    def scenario_dir(self) -> Path:
        return self.root / "scenarios"

    @property
    def output_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def runs_dir(self) -> Path:
        return self.output_dir / "runs"

    @property
    def aggregates_dir(self) -> Path:
        return self.output_dir / "aggregates"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    def validate_required(self) -> None:
        required_dirs = [self.case_dir, self.static_dir, self.scenario_dir]
        for d in required_dirs:
            if not d.exists():
                raise FileNotFoundError(f"필수 디렉토리가 없습니다: {d}")

        # 리빌드 규약: B_41_labeled.csv, gen_result_with_type.csv는 사용하지 않는다.
        required_files = [
            self.case_mfile,
            self.static_dir / "dyn_params.csv",
            # 아래 두 개는 계산 필수는 아니고 메타/검사용
            # (없어도 동작 가능하게 두고 싶으면 required에서 빼도 됨)
            self.static_dir / "bus_location.csv",
            self.static_dir / "line_catalog.csv",
        ]
        for f in required_files:
            if not f.exists():
                raise FileNotFoundError(f"필수 파일이 없습니다: {f}")

    def ensure_output_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.aggregates_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)


PATHS = ProjectPaths(root=find_project_root())


def get_paths(validate: bool = True, ensure_outputs: bool = True) -> ProjectPaths:
    if validate:
        PATHS.validate_required()
    if ensure_outputs:
        PATHS.ensure_output_dirs()
    return PATHS
