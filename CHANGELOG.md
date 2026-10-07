# Project artificial_neuron

## 1. Overview

본 프로젝트의 주 목표는 뉴런의 지속성을 생각하며 최대한 뉴런이 정보를 처리하는 비슷한 알고리즘을 구현시킨 인공지능 모델을 만들어 보는 것이다.

---

## 개발일지

### 26-10-08

ai_rule.md를 수정하였다. 전체적인 개발 단계를 구체적으로 체계화 하였고, 각 단계에서 Architect, Engine의 역할 및 다루는 파일 그리고 세부 진행 흐름을 내용으로 구성했다.

세부적으로는 기존 개발 과정에서는 algorithm_spec.md 문서를 다룰 때 나(Architect)는 알고리즘 자체를 자연어 및 수학으로 최대한 정밀하게 설계하려고 했고, Deepseek(Engine)는 설계에 맞춰 구체적인 프로그래밍 구현 계획을 세웠다. 이 과정에서 문서의 내용이 방대해졌고, Architect가 의도한 부분과 Engine이 의도한 부분이 모호하게 섞이는 내용도 있었다. 따라서 누가 의도한 내용인지 구별하고 향후 수정하는 과정이 더 수월하게 진행되게 하기 위해 이를 algorithm_spec.md, programming_plan.md 문서들로 나누었다.

다른 repository를 구경하며 일반적으로 README.md에는 project 소개와 이를 사용하고 활용하는 방법이 적혀있었다. 하지만 나는 README.md를 project 개요와 개발 일지로 그 내용을 구성했었다. 다른 사람이 README.md를 활용하는 용도와 유사하게 활용하여 향후 다른 사람들이 나의 repository를 구경하러 왔을 때, README.md를 읽으면서 project를 파악하는게 더 익숙하고 쉬운 과정이 되도록 하고싶었다. 따라서 README.md, CHANGELOG.md 문서들의 구성 내용들을 재조립하였다.

마지막으로 git 처리에 관한 부분은 체계화 하지 못하였다. 기존 개발 단계에 없었던 내용이고, 아직 다루지 않은 git의 기능들이 있다고 생각하여 향후 개발을 진행하면서 최적화된 git 버전관리 process를 구상한다면 그 때 반영할 생각이다. 지금은 이전처럼 직접 commit, push 등의 버전 관리를 진행할 계획이다.(github copilot을 활용해 자동화할 수 있는 방법도 고민 중이다.)

---

## 2. v0.0.0 보고서 (26-10-05)

### 2.1 요약

v0.0.0은 프로젝트의 첫 동작 가능한 구현 사이클이다. 인공지능 시스템을 5단계로 구분했을 때 **'정보 처리 알고리즘'** 영역만을 범위로 삼아, 뉴런 유형·전하 동역학·Hebbian 학습·구조 가변성을 갖춘 스파이킹 네트워크를 순수 Python + NumPy로 구현했다. 시뮬레이터, 7종 분석 모듈, 체크포인트, CLI 진입점을 포함하며, 약 158개의 테스트로 회귀를 방지한다.

T=1,000 관찰에서는 설계 목표(활성화 비율 비편향성, 가중치 비포화성)를 만족했지만, T=10,000 장기 관찰에서 **전면 포화 attractor**가 확인되었다. 이는 v0.1.0의 최우선 과제로 이월한다.

---

### 2.2 스코프 및 목표

**시스템 5단계 구분과 프로젝트 범위**

| 단계 | 내용 | v0.0.0 |
| :--- | :--- | :---: |
| 1 | 입력 정보 종류 | — |
| 2 | 임베딩 알고리즘 | — |
| 3 | **정보 처리 알고리즘** | **구현** |
| 4 | 언임베딩 알고리즘 | — |
| 5 | 출력 정보 종류 | — |

**설계 목표 (spec 1.4)**

1. **활성화 비율의 비편향성**: 전체 발화율이 극단(0 또는 1)으로 편향되지 않아야 한다.
2. **가중치 분포의 비포화성**: 시냅스 가중치가 경계값(0 또는 W_max)에 자주 부딪히지 않아야 한다.

**파라미터 조정 원칙**: 초기 구현 → 모니터링 → 상수 조정. 상수만으로 목표 미달 시 알고리즘 자체 수정 검토.

---

### 2.3 아키텍처

**뉴런 모델**

- 3종 뉴런: Input / Internal / Output. 총 N = 20 (Input 2, Internal 17, Output 1).
- E/I 비율: 16 흥분성(+1), 4 억제성(−1). 시드 기반 셔플로 무작위 배정.
- 전하 동역학: 감쇄 + 입력 합산 + 0 클립. 발화 후 리셋.
- 고정 임계값 θ = 1.0.

**핵심 수식**

전하 업데이트 (spec 4.1):

```
V_i(t) = max(0, γ·V_i(t-1)·(1-A_i(t-1)) + Σ A_j^(pre)(t)·S_j·W_e(t-1))
```

- γ = 0.9 (감쇄)
- 입력 뉴런이 pre인 경우 A_j(t), 그 외는 A_j(t-1). (외부 신호의 same-tick 전파)

활성화 (spec 4.2): `A_i(t) = 1 if V_i(t) >= θ else 0`

Hebbian 가중치 업데이트 (spec 5.1):

```
ΔW_e(t) = η_u · R(t) · A_j^(pre)(t) · A_i(t) - η_d · W_e(t-1)
W_e(t) = clip(W_e(t-1) + ΔW_e, 0, W_max)
```

- η_u = 0.05, η_d = 0.01, W_max = 2.0, R(t) = 1.0 (비지도)

구조 가변성 (spec 5.2):

- 소멸: W_e < W_min(=0.01)이면 제거.
- 신생: 100틱 주기, 뉴런별 위상 분산(stagger), source당 최대 1개.
- Out-degree 상한 5개 (자기 연결·중복 포함). 중복 엣지는 독립적으로 학습.

**틱 파이프라인 (spec 3)**

```
Step 1: 입력 뉴런 활성화 갱신 (External Signal Provider)
Step 2: 전 뉴런 V_i(t), A_i(t) 동시 업데이트
Step 3: Hebbian → Pruning → Generation (순서 고정)
Step 4: 출력 벡터 반환
```

**모듈 구조**

```
artificial_neuron/
├── config.py            # 전역 하이퍼파라미터 (frozen dataclass)
├── neuron.py            # Neuron, Edge 데이터 클래스
├── network.py           # 그래프 소유자 (in/out 인덱스, edge_id 발급)
├── input_provider.py    # External Signal Provider 인터페이스 + 기본 순환
├── dynamics.py          # Step 1, 2, 4
├── plasticity.py        # Step 3
├── simulator.py         # tick loop + 프로파일링 + 로그 저장
├── analysis/            # 7종 분석 모듈 (spec 6장)
├── io/checkpoint.py     # 상태 저장/로드
├── tests/               # 약 158개 테스트 (golden 포함)
└── main.py              # CLI 진입점
```

**설계 원칙**

- Network가 그래프의 유일한 소유자. in/out 인덱스 원자적 갱신.
- 엣지는 명시적 edge_id를 가지며 재사용하지 않음.
- 목적별 분리된 3종 RNG 스트림 (init/input/gen). 호출 순서 고정으로 완전 재현.
- 분석 모듈은 저장된 로그만 읽음 (시뮬레이터 독립, spec 6.8).

---

### 2.4 검증 결과

**테스트 현황 (약 158개)**

| 분류 | 대상 | 커버리지 |
| :--- | :--- | :--- |
| 자료구조 | neuron, network, edge_id | in/out 일관성, seed 재현 |
| 알고리즘 | dynamics, plasticity | 손계산 검증, 경계값, 4조합 전수 |
| 시뮬레이터 | tick loop, 로그 저장 | 순서, 프로파일링, 스키마 |
| 분석 모듈 | 7종 | 스모크, CSV 스키마 |
| CLI | main | 인자 검증, analyze 플래그 |
| 회귀 | golden (N=5, T=20) | 완전 스냅샷 비교 |
| 이어하기 | checkpoint | round-trip, 재현 |
| 장기 안정성 | T=10,000 | 관찰 리포트 + 불변식 |

**관찰 리포트 (T=1,000)**

*baseline (Step 3 미포함)*

| 그룹 | 평균 발화율 |
| :--- | :--- |
| input | 0.4000 (외생, 참고) |
| internal | 0.0166 |
| output | 0.0570 |

*완전 루프 (Step 3 포함)*

| 그룹 | 평균 발화율 |
| :--- | :--- |
| input | 0.4000 (외생, 참고) |
| internal | 0.3188 |
| output | 0.2150 |
| pruned events | 135 |
| generated events | 172 |

Step 3 도입으로 internal 발화율이 1.66% → 31.88%로 상승. Hebbian co-fire 강화가 실질적으로 작동함을 확인.

**관찰 리포트 (T=10,000)**

| 지표 | 값 | spec 7.6 참고 범위 | 판정 |
| :--- | :--- | :--- | :--- |
| tail internal+output 발화율 | **0.9444** | < 0.9 | 이탈 (포화) |
| W 경계 포화율 | **0.9184** | < 0.20 | 이탈 (포화) |
| max in-degree | 10 | < N/2 = 10 | 경계 |
| runaway 뉴런 (≥100틱 연속 발화) | **19** | 0 | 이탈 |
| 억제 발화 (tail) | 0.7500 | > 0 | OK (E와 동반 포화) |

---

### 2.5 발견 사항: 전면 포화 attractor

**현상**

T=1,000에서는 정상 범위였던 발화율이 T=10,000에 이르러 **모든 비입력 뉴런이 매 틱 발화하는 포화 평형**으로 수렴. 19개 뉴런 모두가 runaway 상태.

**원인 분석**

Hebbian 평형 조건 `η_u · R · A_j · A_i = η_d · W_e`에서, 지속적 co-fire 시 평형 가중치는 W_e = η_u/η_d = 5.0. 그런데 W_max = 2.0이므로 모든 co-fire 엣지가 상한에 붙잡힘(포화율 91.84%가 이를 반영).

포화 상태에서 뉴런 하나의 in-edge 약 5개 × W_e ≈ 2.0 → net drive ≈ +6.0 → V_ss = 6.0/(1−0.9) = 60.0. 임계값 θ = 1.0에 비해 압도적 초과.

**되먹임 고리**: 모두 발화 → 모든 엣지 co-fire → 모든 W가 W_max로 → drive 증가 → 계속 발화.

현재 파라미터 조합에서 "전면 포화"가 안정적인 attractor로 존재.

**결정 (v0.0.0)**

v0.0.0에서는 **파라미터 조정 없이 관찰 사항으로만 기록**하고, 첫 버전 완성에 집중한다. spec 7.6의 참고 범위는 assert로 강제하지 않으며, 리포트로만 출력한다. 알고리즘 불변식(V_i ≥ 0, 0 ≤ W_e ≤ W_max, 입력 뉴런 in-edge 0개, E/I 비율 보존)만 테스트로 강제한다.

---

### 2.6 v0.1.0 과제 (이월)

**후보 메커니즘**

| 후보 | 내용 | 특징 |
| :--- | :--- | :--- |
| Refractory period | 발화 후 K틱 강제 비활성 | runaway 직접 차단. 생물학적 자연스러움 |
| Incoming normalization | V_i = γ·V·(1−A) + (1/in_degree)·Σ | in-degree 편향 완화 |
| Weight-dependent decay | ΔW = η_u·R·A_j·A_i − η_d·W² | 큰 가중치 강하게 감쇄 |
| V_max cap | V_i = min(V_max, ...) | 단발 안전장치 |
| 파라미터 조정 | θ, W_max, η_u, η_d, γ 조합 | spec 1.4의 1차 전략 |

**진행 방향**: spec 1.4의 원칙에 따라 파라미터 조정을 먼저 시도하고, 상수만으로 해결되지 않으면 알고리즘 수정을 검토한다.

**추가 관찰**

- Input이 (0,0)인 구간에서 두 번째 사이클에도 완전 침묵. 네트워크가 "지속성"을 갖지 못함을 시사. 프로젝트의 핵심 목표와 직결되는 관찰.
- t=300–400 구간에서 output이 (0,0) 입력과 동일하게 0을 출력. 이 시점 네트워크가 두 입력을 구분하지 못함.

---

## 3. 개발일지

* **26-09-29**
    - 이전에 만들어뒀던 코드 베이스가 날아가서 알고리즘을 구성하는 아이디어는 최대한 비슷하게 반영하는 것을 방향성으로 잡아야 할 것 같다.
    - repository created
    - ai_rule.md 생성

* **26-10-01**
    - algorithm_spec.md 초안 작성.
    - 인공지능을 input information, embedding algorithm, information processing algorithm, unembedding algorithm, output information 이렇게 5개의 구성으로 나누었을 때, 현 버전에서 구현할 모델은 information processing algorithm부분 만 구현하는 것으로 그 범위를 분명히 하였다.
    - 모델을 구성하는 알고리즘을 설계하는 단계에서 추가적인 아이디어들이 있었으나 초기 단계에 원할히 구현하는 것, 하드웨어 연산량의 조절을 위해 알고리즘에 도입하지 않은 아이디어도 있었다.

<!--
* **26-10-05** (제안 초안)
    - v0.0.0 첫 구현 사이클 완료.
    - spec 1~7장 구현: config, neuron, network, input_provider, dynamics(Step 1/2/4), plasticity(Step 3), simulator, analysis 7종, io/checkpoint, main.
    - 약 158개 테스트 및 golden regression 통과.
    - T=1,000 관찰에서 설계 목표 만족. T=10,000 관찰에서 전면 포화 attractor 확인.
    - 포화 문제는 v0.1.0의 최우선 과제로 이월. spec 1.4의 "초기 구현 → 모니터링 → 이후 조정" 흐름에 따라 파라미터 조정 사이클에서 다룰 예정.
-->