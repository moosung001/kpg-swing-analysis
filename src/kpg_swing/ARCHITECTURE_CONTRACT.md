# KPG Swing Project Architecture Contract

이 문서는 kpg_swing 프로젝트의 설계 규약을 명문화한다.
모든 구현은 본 문서의 규칙을 우선적으로 따른다.

## 1. Root Anchor 및 경로 규약

- 프로젝트 루트는 `.project_root` 마커 파일이 존재하는 디렉토리로 정의한다.
- `src/kpg_swing/paths.py`는 다음 절대경로를 제공해야 한다.
  - ROOT
  - CASE_DIR = ROOT / "KPG193_ver1_2"
  - STATIC_DIR = ROOT / "data_static"
  - SCENARIO_DIR = ROOT / "scenarios"
  - OUTPUT_DIR = ROOT / "outputs"

원칙:

- 실행 위치(cwd)가 어디든 동일한 결과가 나와야 한다.
- 상대경로(Path("data") 등) 하드코딩은 금지한다.

## 2. 모듈 경계 규약

### 2.1 engine vs metrics

- engine:
  - 입력: 상태 벡터 또는 시계열
  - 출력: 상태 벡터 또는 배열(시계열/라인별 시계열 등)
  - 역할: 물리 계산, 수식, 적분 실행(legacy 포함)
- metrics:
  - 입력: 시계열/배열(trajectory)
  - 출력: scalar 또는 소수의 요약 통계량만
  - 역할: Nadir, RoCoF, Area, lineflow_max, overload_count 등 요약 산출
  - 시계열 전체 저장을 강제하지 않는다.

### 2.2 core의 역할

- core/runner.py는 적분(solve_ivp)을 직접 수행하지 않는다.
- core/runner.py는 engine 호출, 입력 검증, 예외 포맷팅, 요약(metrics 호출)만 수행한다.
- .m 파싱과 무거운 행렬 구성은 core/loader.py에서 1회 수행한다.

## 3. SystemData: Single Source of Truth

`SystemData`는 loader가 1회 생성하며, 이후 모든 계산은 SystemData를 참조한다.
engine이나 runner 내부에서 .m 파일을 재파싱하는 행위는 금지한다.

SystemData v1 최소 필드(권장):

- baseMVA
- gen_buses 또는 gen_bus_idx (동역학 노드 집합)
- B_red 또는 B_41 (동역학 결합 행렬)
- M, D (관성, 감쇠)
- Pm0 또는 초기 입력 파라미터
- optional: lineflow 계산에 필요한 Bbus, lines, bus_id_map 등

추가 필드는 v2 확장으로 취급하고, v1을 깨지 않도록 한다.

## 4. Scenario Schema 규약

배치 입력은 CSV 또는 JSON으로 저장한다.
CSV 기준 최소 컬럼은 다음을 포함해야 한다.

필수:

- scenario_id: 문자열 또는 정수, 유일해야 함
- disturb_bus: 외란 위치(버스 번호 또는 동역학 노드 인덱스)
- disturb_type: 예) step_Pm, step_Pe 등 문자열
- magnitude: 외란 크기
- t_start: 외란 시작 시간 [s]
- t_end: 외란 종료 시간 [s]

단위 규약:

- magnitude의 단위를 명시해야 한다.
  - MW 기반이면 metadata에 baseMVA 변환 규칙을 기록한다.
  - pu 기반이면 pu로 고정한다.

## 5. Output Storage Protocol

배치 실행 결과는 반드시 다음 3종 세트를 포함한다.

outputs/runs/{run_id}/

- metadata.json
  - SimConfig, RunConfig
  - 코드 버전 또는 git commit(가능하면)
  - 입력 시나리오 파일명
- scenario_table.csv
  - 실제 실행한 시나리오 테이블(정렬/필터링 반영 후 최종본)
- results_summary.parquet (또는 csv)
  - scenario_id를 키로 하는 요약 결과 테이블
  - 최소 포함 권장: nadir, rocof_max, area, success_flag, error_message
  - lineflow 옵션 사용 시: lineflow_max, overload_count, worst_line_id 등

원칙:

- 시계열 원본 저장은 기본 비활성화.
- 필요 시 대표 시나리오만 별도 옵션으로 저장하도록 한다.

## 6. Parallel Execution Protocol

- 병렬 실행 시 SystemData를 작업마다 pickle로 전달하는 방식을 피한다.
- 권장 방식:
  - 프로세스별 initializer에서 loader를 1회 호출해 SystemData를 프로세스 로컬 캐시에 보관
  - 각 워커는 scenario_row만 받아 runner를 호출

## 7. Determinism 및 로그 규약

- 난수(샘플링)가 들어가는 경우 seed를 RunConfig에 포함하고 metadata.json에 기록한다.
- runner는 예외 발생 시 프로세스를 종료하지 않고,
  - success_flag = false
  - error_message에 요약된 에러를 기록
  - 나머지 시나리오는 계속 진행

## 8. Legacy Preservation Rule

engine/ 폴더의 레거시 파일은 파일명과 로직을 최대한 유지한다.
허용되는 변경:

- 경로 하드코딩 제거: paths.py 사용
- 불필요한 print, plotting, UI 관련 코드 제거 또는 비활성화
- lineflow 계산은 옵션화(compute_lineflow 같은 플래그)

금지되는 변경(초기 단계):

- 레거시 엔진을 여러 파일로 쪼개는 대규모 리팩토링
- solve_ivp 설정을 근거 없이 변경
