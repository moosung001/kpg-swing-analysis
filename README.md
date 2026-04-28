# KPG-193 Swing Simulator v2

> **논문 서밋 버전** — 전력계통 주파수 응답의 네트워크 중심성 기반 취약 버스 분석

한국 전력망(KPG-193, 193버스) 모델에 스윙 방정식을 적용한 배치 시뮬레이터입니다.
발전기 탈락 시나리오 대규모 배치 실행 → 각 버스의 주파수 취약도(Nadir, RoCoF) 계산 → 네트워크 중심성 지표와의 상관관계 분석까지 수행합니다.

## 핵심 기술

- **Kron Reduction**: 193버스 → 41 발전기버스 축소 (내부 구현, PowerModels 의존 없음)
- **Swing Equation ODE**: SciPy RK45, 단일/다중 외란 지원
- **배치 시뮬레이션**: 모든 발전기 버스 × 복수 ΔP 조합 병렬 실행
- **취약도 분석**: Nadir, RoCoF, Angle Spread 등 지표 산출
- **중심성 상관관계**: PageRank, Closeness, Betweenness, Eigenvector vs. 취약도 지표
- **Pareto 프런트**: 관찰 버스 선정 최적화 (취약도 커버리지 vs. 버스 수)

## 주요 결과

\![Pareto Analysis](fig_pareto_masterpiece_final_band.png)

## 프로젝트 구조

```
├── src/kpg_swing/          # 핵심 패키지
│   ├── engine/             # 물리 계산 엔진
│   │   ├── swing_api.py    # Swing ODE API
│   │   ├── events.py       # 외란 이벤트 처리
│   │   ├── internal_kron.py # Kron 축소
│   │   ├── dcflow.py       # DC 파워플로우
│   │   ├── bus_restore.py  # Kron 역복원
│   │   └── islanding.py    # 고립 섬 처리
│   ├── core/
│   │   ├── loader.py       # SystemData 로더
│   │   ├── metrics.py      # Nadir, RoCoF 등 지표 계산
│   │   └── checks.py       # 입력 검증
│   ├── paths.py            # 경로 관리
│   └── ARCHITECTURE_CONTRACT.md  # 설계 규약 문서
│
├── scripts/                # 배치 실행 & 분석 스크립트
│   ├── main_batch_step.py  # 배치 시뮬레이션 실행 진입점
│   ├── analyze_batch_step.py # 결과 분석
│   ├── analyze_network_vs_response.py  # 중심성 상관관계
│   └── ...                 # 시각화 스크립트
│
├── data_static/            # 정적 입력 데이터
│   ├── dyn_params.csv      # 발전기 동특성 (H, M, D)
│   ├── bus_location.csv    # 버스 지리 좌표
│   └── line_catalog.csv    # 선로 정격 데이터
│
├── KPG193_ver1_2/          # 한국 전력망 모델 (MATPOWER 형식)
├── scenarios/              # 시나리오 입력 CSV
├── pyproject.toml
└── plot_coi_paper.py       # 논문 그림 재현 스크립트
```

## 실행 방법

```bash
pip install -e .
pip install scipy matplotlib networkx

# 배치 시뮬레이션 실행
python scripts/main_batch_step.py

# 결과 분석
python scripts/analyze_batch_step.py
python scripts/analyze_network_vs_response.py
```

## 기술 스택

`Python` `NumPy` `SciPy` `Pandas` `NetworkX` `Matplotlib` `Seaborn`
