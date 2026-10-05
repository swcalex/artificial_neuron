# Algorithm Specification: artificial_neuron

본 문서는 `artificial_neuron` 모델의 핵심 구성 요소 및 수학적·논리적 명세를 정의하는 문서이다. Architect와 Engine 간 논의된 모든 핵심 아이디어와 수식을 총망라하여 모델 구현 및 유지보수의 단일 진실 공급원(Single Source of Truth) 역할을 한다.

---

## 1. 모델 개요 및 범위 (Overview & Scope)

### 1.1 시스템 구성 내 스코프
인공지능 일반 시스템을 5가지 단계로 구분할 때, 본 프로젝트는 **'정보 처리 알고리즘'** 영역에 한정하여 설계한다.

1. 입력 정보 종류 (Input Data Types)
2. 임베딩 알고리즘 (Embedding Algorithm)
3. **정보 처리 알고리즘 (Information Processing Algorithm)** $\leftarrow$ **\[본 프로젝트 영역\]**
4. 언임베딩 알고리즘 (Unembedding Algorithm)
5. 출력 정보 종류 (Output Data Types)

### 1.2 핵심 동작 특성
* **지속성 및 동시성**: 추론(Inference)과 학습(Learning)이 상호 분리되지 않고 작동 중에 동시에 지속적으로 진행된다.
* **시간 구동형 시뮬레이션**: 매 타임스텝(Tick)마다 외부 자극 입력 $\rightarrow$ 내부 전하 전달 및 전하값 업데이트 $\rightarrow$ 가중치 학습/시냅스 구조 변경 $\rightarrow$ 출력 뉴런 반응의 순환 체계로 작동한다.
* **공간 좌표 배제 (연산 최적화)**: 연산 효율성을 위해 3차원 물리적 위치 매핑을 배제하고, 가중치 그래프 구조(Adjacency Representation)를 사용한다.

### 1.3 시간 단위 및 틱 정의 (Tick Definition)
* **논리적 이산 시간**: 1틱(Tick)은 wall-clock 상의 고정된 초 단위로 정의하지 않는다. 틱은 모델 내부의 논리적 이산 시간 단위이며, 모든 감쇄율·학습률·구조 변화 주기는 "1틱당" 값을 기준으로 정의한다.
* **틱 인덱스**: 틱은 $t = 0$부터 시작한다. 최초 틱 진입 시 이전 틱 상태는 $A_i(-1) = 0$ (전 뉴런), $V_i(-1) = V_i(0)$ (초기 전하값), $W_e(-1) = W_{\text{init}}$으로 정의한다.
* **이유**: 현재 수식의 $\gamma$, $\eta_u$, $\eta_d$ 등은 모두 per tick 상수이다. 실제 연산 속도나 하드웨어 부하에 따라 물리 시간이 달라지더라도, 모델의 동역학과 재현성은 틱 인덱스 기준으로 유지되어야 한다.
* **런타임 모니터링**: 모델 가동 시 다음 지표를 기록한다.
  * `tick_duration_ms`: Step 1~4 전체 연산에 걸린 실제 시간
  * `ticks_per_second`: 초당 처리 틱 수
  * `step_profiling`: Step 1/2/3/4 각각의 소요 시간
  * `logical_tick`과 `wall_clock_time`의 대응 로그
* **실시간 상호작용 확장**: 추후 외부 환경과 실시간 상호작용이 필요해지면, 별도 스케줄러 계층에서 목표 틱 레이트(예: `target_tick_hz`)를 정의하고 skip/backpressure 정책을 둔다. 이는 현재 정보 처리 알고리즘 스코프 외부의 런타임 계층으로 분리한다.

### 1.4 설계 목표 및 파라미터 조정 원칙 (Design Goals)
본 모델의 초기 파라미터는 다음 두 가지 목표를 만족하도록 모니터링 기반으로 조정한다.

1. **활성화 비율의 비편향성**: 전체 뉴런의 발화 비율이 극단(0 또는 1)으로 편향되지 않아야 한다. 출력 정보가 지나치게 단조로워지는 것을 방지한다.
2. **가중치 분포의 비포화성**: 시냅스 가중치가 경계값($0$ 또는 $W_{\max}$)에 자주 부딪히지 않고, 그 사이 범위에서 적절히 조절되어야 한다.

**조정 전략**: 우선 현재 수식과 상수·초기화 값으로 초기 구현을 진행하고, 지속적인 모니터링을 통해 파라미터를 조정한다. 상수·초기화 조정만으로 위 목표를 달성하지 못할 경우에 한해 알고리즘 자체의 수정 또는 추가 메커니즘 도입을 검토한다.

---

## 2. 뉴런 유형 및 구조 (Neuron Types & Data Structure)

### 2.1 뉴런 유형 분류
네트워크 내부 뉴런은 역할에 따라 3가지로 분류한다.

* **Input Neuron (입력 뉴런)**: 매 틱마다 외부 자극/자율 신호 형태로 활성화 정보($0$ 또는 $1$)를 수신.
   * 신호 변질 방지 제약: 외부 입력 신호의 오염 및 변질을 막기 위해 타 뉴런으로부터 입력 신호를 받는 연결(In-Edges) 형성을 전면 금지한다. 전하 축적 과정 없이 활성화 상태만 수신 및 전파한다.
   * 입력 신호 공급원: 입력 뉴런의 활성화 상태 $A_{\text{input}}(t)$는 외부 신호 공급자(External Signal Provider)로부터 매 틱 공급받는다. 기본 공급자는 아래 2.4에 정의된 순환 패턴을 사용하며, 추후 실환경 모듈로 교체 가능하도록 인터페이스를 분리한다.
   * Out-Edge: 다른 뉴런(내부/출력)과 동일하게 자유롭게 형성 가능하다. Input Neuron을 In-Edge의 타깃으로 삼는 것만 제약이며, Input Neuron이 스스로 Out-Edge를 생성하는 것은 허용한다.
* **Internal Neuron (내부/중계 뉴런)**: 신호 수신, 전하 축적, 역치 도달 시 발화 및 타 뉴런으로 신호 전파 담당.
* **Output Neuron (출력 뉴런)**: 최종 정보 처리 결과로서의 활성화 상태 모음(State Vector)을 외부로 출력.
   * 연결 확장성: 신호를 받는 것뿐만 아니라 필요시 타 뉴런(입력 뉴런 제외)으로 연결 시냅스(Out-Edges)를 형성하여 신호를 보낼 수 있다.

### 2.2 뉴런 속성 명세 (Neuron Attributes)

| 속성 (Attribute) | 데이터 타입 / 형태 | 변수/상수 | 기본 설정값 / 범위 | 설명 |
| :--- | :--- | :--- | :--- | :--- |
| **고유번호 (ID)** | `Integer` | **상수** | `0 ~ N-1` | 네트워크 내 뉴런 식별 고유 ID |
| **고유 전달 강도 (Intrinsic Strength)** | `Integer` | **상수** | $+1$ (흥분성) / $-1$ (억제성) | 뉴런 고유의 기본 신호 전달 성질 ($S_i$) |
| **전하값 (Charge / Voltage)** | `Float` | **변수** | 초기값 $V_i(0) \sim \mathcal{N}(0.1, 0.05)$ ($V_i \ge 0$) | 뉴런 내부 축적 상태값 ($V_i$), 입력 뉴런은 보유하지 않음 |
| **활성화 정보 (Activation State)** | `Binary` | **변수** | $0$ 또는 $1$ | $0$: 비활성화 / $1$: 발화 상태 ($A_i$) |
| **발화 임계값 (Threshold)** | `Float` | **상수** | $\theta_i = 1.0$ (모든 비입력 뉴런 고정) | 발화 판정 기준값. 입력 뉴런은 보유하지 않음 |
| **연결 리스트 (Out-Edges)** | `List[(Edge ID, Target ID, Weight)]` | **변수** | 최대 5개 제한 | 신호를 전송할 대상 뉴런 ID 및 가중치 목록. 자기 연결 및 동일 타깃 중복 연결 허용 |

* **연결 의미론**: 동일한 `(Source ID, Target ID)` 쌍에 여러 엣지가 존재할 수 있으며, 각 엣지는 독립적인 시냅스로 취급한다. 따라서 학습, 감쇄, 소멸, 신호 합산은 각 엣지 단위로 수행한다.
* **Input Neuron 제약**: Input Neuron은 In-Edges를 받을 수 없으므로, 어떤 시냅스 생성에서도 타깃이 Input Neuron이 될 수 없다.
* **임계값 지위**: $\theta_i$는 학습으로 변화하지 않는 상수이며, 모든 비입력 뉴런에 대해 $\theta_i = 1.0$으로 고정한다. 추후 파라미터 조정 사이클에서 뉴런별 또는 전역 상수 조정이 필요해지면 재검토한다.

### 2.3 초기화 전략 (Initialization Strategy)

* **기본 네트워크 규모**: $(N_{\text{input}}, N_{\text{internal}}, N_{\text{output}}) = (2, 17, 1)$, 총 $N = 20$.
* **ID 매핑 (고정)**:
  * ID `0, 1`: Input Neuron
  * ID `2 ~ 18`: Internal Neuron (17개)
  * ID `19`: Output Neuron
* **E/I 비율**: 전체 뉴런의 80%는 흥분성($S_i = +1$), 20%는 억제성($S_i = -1$)으로 초기 배정하여 네트워크 균형 유지.
  * $N = 20$ 기준: 흥분성 16개, 억제성 4개.
  * 배정 방식: RNG 기반. ID 목록을 무작위로 셔플한 뒤 앞에서부터 4개를 억제성으로 지정한다 (나머지 16개는 흥분성). 이 방식은 억제성 개수와 무작위성 모두를 보장하며, 재현 가능하다.
* **전하 초기화**: 입력 뉴런을 제외한 모든 내부/출력 뉴런의 초기 전하값은 $V_i(0) \sim \mathcal{N}(0.1, 0.05)$로부터 샘플링한다. 샘플링된 값이 음수일 경우 **클립(clip) 후처리**를 적용하여 $V_i(0) = \max(0, \text{sample})$로 확정한다. 재현성과 결정성을 위해 재샘플링이나 절댓값 변환은 사용하지 않는다. 이는 초기 편향을 0으로 고정하지 않음으로써, 첫 틱에서 극소수 뉴런이 발화하여 부트스트랩을 시드할 가능성을 열어두기 위함이다.
* **시냅스 초기화**: **희소 무작위 연결(Sparse Random Initialization)** 방식 채택.
  * 부트스트래핑 지연을 방지하기 위해 각 뉴런은 초기 생성 시 3개의 무작위 대상 뉴런과 낮은 초기 가중치($W_{\text{init}} = 0.2$)로 연결 시작.
  * 초기 시냅스 생성 절차:
    1. 뉴런 ID 오름차순으로 순회한다 (`0, 1, ..., N-1`). Input Neuron도 Out-Edge 생성 대상에 포함된다.
    2. 각 뉴런마다 3회 반복하여 타깃을 샘플링한다.
    3. 타깃 후보 집합은 **Input Neuron을 제외한 모든 뉴런**(ID `2 ~ N-1`)이다.
    4. 매 시도마다 후보 집합에서 uniform random으로 하나를 뽑는다. 중복 타깃과 self-loop가 나와도 그대로 수용한다.
    5. 유효하지 않은 후보(즉 Input Neuron)는 애초에 후보 집합에서 제외하므로 재추첨이 필요 없다.
* **RNG 구성 및 호출 순서 (재현성 보장)**:
  * 단일 RNG 스트림이 아닌, 목적별로 분리된 **명명된 RNG 묶음**을 사용한다. 각 스트림은 자체 seed를 가지며, 상태 전체를 `run_meta.json` / 체크포인트에 저장한다.
  * 스트림 목록:
    * `init_rng`: E/I 배정, $V_i(0)$ 샘플링, 초기 시냅스 타깃 샘플링
    * `input_rng`: 확률적 입력 공급자 사용 시(기본 공급자는 결정적 순환이므로 미사용)
    * `gen_rng`: Step 3의 시냅스 신생 타깃 샘플링
  * **호출 순서 (고정)**:
    1. `init_rng`로 E/I 배정 (ID 목록 셔플 후 앞 4개 억제성)
    2. `init_rng`로 $V_i(0)$ 샘플링 (ID `2 ~ N-1` 오름차순, 각 1회)
    3. `init_rng`로 초기 시냅스 생성 (ID `0 ~ N-1` 오름차순, 각 뉴런당 3회 타깃 샘플링)
  * 이 순서를 고정함으로써 seed가 동일하면 완전 재현이 보장된다.

### 2.4 기본 외부 입력 공급자 (Default External Signal Provider)

* **기본 모드 (결정적 순환)**: 매 200틱마다 입력 패턴이 다음 순서로 순환한다.
  * `pattern_index = (t // 200) mod 4`
  * `pattern[0] = (A_0, A_1) = (0, 0)`
  * `pattern[1] = (0, 1)`
  * `pattern[2] = (1, 0)`
  * `pattern[3] = (1, 1)`
  * 즉, 첫 200틱은 `(0,0)`, 다음 200틱은 `(0,1)`, 그 다음 `(1,0)`, 그 다음 `(1,1)`, 이후 반복.
* **확장성**: 외부 신호 공급자는 인터페이스(예: `provider.sample(t) -> tuple[int, ...]`)로 추상화하여, 추후 임의의 실환경 신호 또는 확률적 공급자로 교체 가능하게 한다.
* **재현성**: 기본 공급자는 결정적이므로 `input_rng`를 사용하지 않는다. 확률적 공급자를 사용할 경우 `input_rng` 상태를 반드시 기록한다.

---

## 3. 타임스텝(Tick) 단위 동작 파이프라인

매 틱($t$) 마다 순차적으로 실행되는 연산 로직은 다음과 같다.

```
[Step 1] 입력 뉴런 활성화 업데이트 (External Signal Provider → A_input(t))
│
▼
[Step 2] 전 뉴런 전하값 V_i(t) 및 활성화 상태 A_i(t) 계산
│
▼
[Step 3] 가중치 업데이트(Three-Factor Hebbian) 및 시냅스 구조 변경 (Pruning / Generation)
│
▼
[Step 4] 출력 뉴런의 활성화 정보 묶음(Output Vector) 반환
```

### 3.1 Step 세부 규약

* **Step 1 (입력 갱신)**: External Signal Provider로부터 길이 $N_{\text{input}}$의 이진 벡터를 받아 입력 뉴런의 $A_{\text{input}}(t)$에 할당한다. 입력 뉴런은 전하를 갖지 않으므로 $V$ 업데이트 대상이 아니다.
* **Step 2 (동시 업데이트 원칙)**: 모든 비입력 뉴런의 $V_i(t)$와 $A_i(t)$는 오직 이전 틱의 상태($V(t-1)$, $A(t-1)$, $W(t-1)$)만을 사용해 계산한다. 계산 도중 일부 뉴런의 $A_i(t)$가 먼저 확정되더라도, 그 값을 같은 틱의 다른 뉴런 전하 계산에 사용하지 않는다. (입력 뉴런에 대한 유일한 예외는 4.1 참조.)
* **Step 3 (구조 가변)**: 세부 순서는 5.4를 따른다.
* **Step 4 (출력)**: Step 2에서 확정된 Output Neuron의 활성화 상태를 ID 오름차순으로 묶어 반환한다. 반환 shape은 `(N_output,)`이며, 현재 설정에서는 스칼라 1개 벡터다.

---

## 4. 수학적/논리적 수식 명세 (Algorithm Specifications)

### 4.1 전하값 업데이트 수식 ($V_i(t)$ Update Function)

뉴런 $i$의 타임스텝 $t$에서의 전하값 $V_i(t)$는 이전 전하값의 감쇄, 발화 후 초기화(Reset), 이전 틱에서 활성화된 이웃 뉴런들로부터 수신된 신호의 합, 그리고 하방 발산 방지를 위한 0 클립으로 계산된다.

$$
V_i(t) = \max\left(0,\ \gamma \cdot V_i(t-1) \cdot (1 - A_i(t-1)) + \sum_{e = (j \to i)} A_j^{(\text{pre})}(t) \cdot S_j \cdot W_e(t-1)\right)
$$

* $\gamma = 0.9$: 전하 자연 감쇄율 (Decay Factor)
* $A_i(t-1) \in \{0, 1\}$: 이전 틱의 발화 여부 (발화 시 전하 $0$으로 리셋)
* $S_j \in \{+1, -1\}$: 사전 뉴런 $j$의 고유 전달 강도
* $e = (j \to i)$: 뉴런 $j$에서 뉴런 $i$로 향하는 개별 시냅스 엣지
* $W_e$: 해당 엣지의 가중치
* $A_j^{(\text{pre})}(t)$: 사전 뉴런 $j$의 활성화 상태로, 다음 규칙에 따라 결정된다.
  * 만약 $j$가 Input Neuron이면 $A_j(t)$ (Step 1에서 갱신된 당 틱 값)를 사용한다.
  * 그렇지 않으면 $A_j(t-1)$을 사용한다.
* 합은 뉴런 $i$로 들어오는 모든 시냅스 엣지에 대해 수행한다. 자기 연결($i \to i$)도 포함될 수 있으며, 동일한 $j \to i$ 쌍에 중복 엣지가 있으면 각 엣지를 별도로 합산한다.

**0 클립의 설계 의도 (Rationale)**:
전하값 $V_i$가 음의 방향으로 무한히 발산하는 것을 방지하기 위해 $\max(0, \cdot)$ 클립을 적용한다. 이는 가중치 $W_e$를 $[0, W_{\max}]$로 제한하는 것과 동일한 성격의 안전장치다.
* 억제성 사전 뉴런($S_j = -1$)의 발화로 인해 합산 항이 음수가 될 수 있고, 감쇄 항이 $0$에 가까워지면 $V_i(t)$가 음수가 될 수 있다.
* 클립이 없으면 억제성 입력이 지속될 때 $V_i$가 음의 방향으로 표류하여, 이후 흥분성 입력이 들어와도 임계값 $\theta_i$에 도달하기까지 비정상적으로 큰 양의 입력을 필요로 하게 된다 (hysteresis 유사 현상).
* 클립을 적용하면 $V_i = 0$에서 하한이 고정되어, 흥분성 입력이 들어오는 즉시 표준적인 축적 동역학이 재개된다.

**입력 뉴런 예외의 정당성**: 입력 신호는 외생적으로 매 틱 공급되는 신호이므로, 공급된 당 틱에 즉시 하류 뉴런의 전하 계산에 반영되어야 한다. 이는 "외부 자극 입력 → 내부 전하 전달"이 동일 사이클 내에서 이루어진다는 1.2의 서술과 일치한다. 비입력 뉴런 간의 상호작용은 여전히 엄격한 이전 틱 상태만을 사용한다.

### 4.2 활성화 조건 방정식 ($A_i(t)$ Activation Function)

뉴런 $i$의 활성화 상태 $A_i(t)$는 계산된 전하값 $V_i(t)$가 설정된 임계값 $\theta_i$ 이상일 때 $1$(발화)로 결정된다.

$$
A_i(t) = \begin{cases} 1 & \text{if } V_i(t) \ge \theta_i \\ 0 & \text{if } V_i(t) < \theta_i \end{cases}
$$

* $\theta_i = \theta_0 = 1.0$: 모든 비입력 뉴런에 대해 고정 (2.2 참조).

---

## 5. 학습 및 구조적 가변성 명세 (Learning & Plasticity Rules)

### 5.1 3-요소 헵 학습 규칙 (Three-Factor Hebbian Rule)

사전 뉴런과 사후 뉴런의 동시 활성화 기여도에 전역 보상 신호 $R(t)$를 곱하여 가중치 업데이트를 수행한다.

$$
\Delta W_e(t) = \eta_{u} \cdot R(t) \cdot A_j(t-1) \cdot A_i(t) - \eta_{d} \cdot W_e(t-1)
$$

$$
W_e(t) = \min\left(W_{\max}, \max(0, W_e(t-1) + \Delta W_e(t))\right)
$$

* $e = (j \to i)$: 뉴런 $j$에서 뉴런 $i$로 향하는 개별 시냅스 엣지
* $\eta_u = 0.05$: 강화 학습률 (Strengthening Rate)
* $\eta_d = 0.01$: 자연 감쇄율 (Decay Rate)
* $R(t)$: 전역 보상 신호. 비지도 학습 환경 시 $R(t) = 1.0$으로 고정 동작.
* $W_{\max} = 2.0$: 가중치 상한선 (폭주 방지 정규화)
* 중복 엣지가 존재하는 경우, 각 엣지는 독립적으로 위 업데이트를 적용받는다.

**사전 뉴런 활성화 시점 (A_j^(pre)(t)) — 4.1과의 통일**:
Hebbian 수식의 co-fire 항에 들어가는 사전 뉴런 활성화는 dynamics 4.1과 동일한 규칙을 따른다.
* 사전 뉴런 $j$가 Input Neuron이면 $A_j(t)$ (Step 1에서 갱신된 당 틱 값)를 사용한다.
* 그 외의 $j$는 $A_j(t-1)$을 사용한다.

이 통일은 두 가지 근거에 따른다.
1. **인과 정합성**: dynamics에서 input pre는 same-tick으로 하류에 전파된다. Hebbian이 만약 input pre에 $A_j(t-1)$을 사용하면, "t-1의 입력과 t의 post 발화"를 짝지어 학습하게 되어 실제 인과가 1틱 어긋난다.
2. **일관성**: Step 2와 Step 3가 서로 다른 시점의 사전 활성화를 참조하면, 두 Step이 서로 다른 동역학을 가정하는 셈이 된다.

비-input pre는 Step 2가 A를 t로 덮어쓰기 전에 캡처된 $A_j(t-1)$을 사용하므로, Simulator가 매 틱 Step 1 이전에 `prev_A` 스냅샷을 만들고 이를 Step 3에 전달한다.

**억제성 시냅스의 Hebbian 부호에 대한 설계 의도 (Rationale)**:
사전 뉴런 $j$가 억제성($S_j = -1$)일 때, 위 수식은 $S_j$를 곱하지 않는다. 즉 억제성 뉴런의 시냅스도 흥분성 뉴런과 동일하게 co-activation 시 강화된다. 이는 의도된 설계이다.
* 억제성 pre $j$의 발화로 post $i$의 전위가 억제되었고, 그럼에도 불구하고 post $i$가 발화했다면, 이는 $j$의 발화가 $i$의 활성화에 "유의미한 영향"을 미쳤음을 시사한다.
* 이때 $W_e$가 증가하면, 이후 $S_j \cdot W_e = -W_e$의 크기가 커져 $i$에 대한 억제 효과가 더 강해진다.
* 만약 $i$가 충분히 억제되어 발화를 멈추면 Hebbian 조건($A_i = 1$)이 깨져 $W_e$는 자연 감쇄로 되돌아간다. 즉, 억제 시냅스 강화는 **negative feedback 홈오스타시스**로 작동한다.
* 만약 $S_j$를 가중치 업데이트에도 곱한다면, 억제성 시냅스는 co-activation 시 오히려 약화되어 "억제 내성"이 생기는 반대 효과를 낳는다. 이는 본 모델의 억제 메커니즘과 상충하므로 채택하지 않는다.

### 5.2 시냅스 사멸 및 신생 조건 (Structural Plasticity)

* **소멸 (Pruning)**: 개별 시냅스 엣지의 가중치 $W_e$가 최소 임계값 $W_{\min}$ 미만으로 감소하면 해당 연결을 `Out-Edges`와 `In-Edges` 인덱스에서 완전 제거한다.
  * $W_{\min} = 0.01$
  * 소멸 조건: $W_e < W_{\min}$
  * $W_{\text{init}} = 0.2$, $\eta_d = 0.01$일 때, 사용되지 않는 신규 시냅스는 대략 299틱 후 $W_{\min}$에 도달하여 소멸한다.
* **신생 (Generation)**:
  * 기본 생성 주기: $T_{\text{gen}} = 100$ ticks
  * 뉴런별 분산(stagger) 적용: 뉴런 $i$는 $t \bmod T_{\text{gen}} = i \bmod T_{\text{gen}}$인 틱에서만 생성 검사를 수행한다.
  * 생성 조건:
    1. 해당 뉴런의 `Out-Edges` 개수가 최대 제한(5개) 미만일 것. 이 5개는 self-loop과 중복 엣지를 모두 포함한 개수다.
    2. 타깃이 Input Neuron이 아닐 것 (Input Neuron은 In-Edges 금지). **타깃 후보 집합은 `2 ~ N-1`이며, 이로써 self-loop은 Input Neuron이 소스일 때 자동 배제된다.**
    3. 자기 자신을 타깃으로 하는 연결 허용 (단, 소스가 Input Neuron인 경우는 후보 집합 규칙에 의해 자연히 배제됨)
    4. 동일 타깃에 대한 중복 연결 허용
    5. 한 생성 이벤트당 최대 1개의 신규 시냅스 생성
  * **타깃 선택 절차**:
    1. 후보 집합 `C = {2, 3, ..., N-1}` (Input Neuron 제외)
    2. `gen_rng`로 `C`에서 uniform random으로 1개 추출
    3. 추출된 타깃으로 신규 엣지 1개 생성
    * 후보 집합이 후보 필터를 이미 통과한 집합이므로 재추첨 절차는 필요하지 않다. 재추첨이 필요한 상황(예: 후보 집합이 비어 있음)이 발생하면 신규 생성을 건너뛴다.
  * 신규 시냅스 초기 가중치: $W_{\text{init}} = 0.2$
  * 중복 연결이 생성된 경우, 각 엣지는 독립적인 시냅스로 학습·감쇄·소멸된다.
* **신규 엣지의 학습 개시 시점**: Step 3에서 생성된 신규 엣지는 **다음 틱($t+1$)부터** Hebbian 업데이트 대상이 된다. 생성 당 틱에는 $\Delta W$ 계산에 포함되지 않는다 (5.4의 순서에 따름).

### 5.3 항상성 메커니즘 (Homeostasis)

* **E/I 균형**: 흥분성 뉴런(80%)과 억제성 뉴런(20%)의 고정 비율로 네트워크 과활성화 및 전면 사멸 방지.
* **가중치 상한 ($W_{\max}$)**: 가중치 무한 발산을 물리적으로 제어.
* **억제성 시냅스의 Negative Feedback**: 5.1의 설계 의도에 따라 억제 시냅스는 co-activation 시 강화되지만, 과억제로 post 발화가 멈추면 자연 감쇄한다.

### 5.4 Step 3 내부 실행 순서 (Step 3 Execution Order)

Step 3은 다음 순서로 실행한다.

1. **Hebbian 업데이트 (기존 엣지 전량)**: 틱 $t$ 진입 시점에 존재하는 모든 엣지 $e$에 대해 5.1 수식을 적용하여 $W_e(t)$를 확정한다. 이 단계에서 새로 계산된 $W_e(t)$는 이후 pruning 판정의 입력이 된다.
2. **Pruning**: 1단계에서 확정된 $W_e(t)$가 $W_{\min}$ 미만인 엣지를 `Out-Edges` 및 `In-Edges` 인덱스에서 제거한다. 제거는 각 엣지 단위로 독립적으로 수행한다.
3. **Generation**: 5.2의 stagger 조건을 만족하는 뉴런에 대해 최대 1개의 신규 엣지를 생성한다. 생성된 엣지는 `In-Edges`/`Out-Edges` 양쪽에 등록한다.

**원칙**:
* 위 1→2→3의 순서는 고정이며, 서로 다른 단계의 조작이 뒤섞이지 않는다.
* 1단계에서 계산된 $W_e(t)$는 2단계에서 소멸되더라도 최종 상태 로그에는 반영되지 않는다 (엣지 자체가 사라지므로).
* 3단계에서 생성된 신규 엣지는 같은 틱의 1단계 Hebbian 계산에 소급되지 않는다.
* 이 순서 규약으로 인해 Step 3는 Step 2의 결과($A(t)$, $V(t)$)에만 의존하며, Step 3 자체의 중간 결과가 Step 2의 재계산을 유발하지 않는다.

---

## 6. 검증 및 시각화 프레임워크 (Verification & Analysis)

모델 구현 후 성능 및 작동 타당성을 평가하기 위한 분석 도구들을 정의한다. 각 도구는 `analysis/` 하위 모듈로 분리 구현되며, 시뮬레이터가 생성하는 로그·상태 스냅샷을 입력으로 받아 독립적으로 실행 가능해야 한다. 모든 산출물은 `outputs/` 디렉터리에 저장한다.

> **비고 (분석 명세의 유예)**: 본 장의 세부 파라미터(스냅샷 주기, top-K, t-SNE perplexity, modularity 알고리즘, `class_separation_score` 정의 등)는 구현 과정에서 실제 데이터 형태와 연산 비용을 보면서 확정한다. 확정된 값은 본 문서에 반영하여 갱신한다.

### 6.1 뉴런 활성화 분포 시각화 (Activation Heatmap)

* **목적**: 전체 뉴런의 발화 비율 및 시간 흐름에 따른 E/I 균형 모니터링. 1.4의 "활성화 비율 비편향성" 목표를 검증한다.
* **입력**: 매 틱의 활성화 상태 벡터 $A(t) \in \{0,1\}^N$, 각 뉴런의 $S_i$ (E/I 라벨).
* **출력**:
  * `activation_heatmap.png`: x축 tick, y축 뉴런 ID, 색상 $A_i$의 히트맵
  * `firing_rate.png`: 틱별 전체 발화율 시계열 곡선
  * `firing_rate_ei.png`: 흥분성/억제성 발화율을 분리한 시계열 곡선
  * `activation_summary.csv`: `tick, total_firing_rate, exc_firing_rate, inh_firing_rate`

### 6.2 시냅스 가중치 인접 행렬 및 그래프 (Adjacency Matrix & Graph)

* **목적**: 학습 진행에 따라 네트워크가 Small-world 구조나 모듈화 클러스터로 자율 구조화되는지 관찰한다.
* **입력**: 측정 주기별 전체 엣지 스냅샷 `(edge_id, source, target, weight)`.
* **출력**:
  * `adjacency_matrix_t{tick}.png`: $N \times N$ 인접 행렬 히트맵. 중복 엣지는 동일 셀에 **합산**하여 표시한다.
  * `graph_t{tick}.png`: networkx 기반 방향 그래프 시각화 (노드 색상=E/I, 엣지 두께=가중치)
  * `graph_metrics.csv`: `tick, avg_degree, clustering_coefficient, modularity, edge_count`
  * (선택) 위 프레임들을 묶은 `graph_evolution.gif`

### 6.3 동적 결정 경계 투영 (Dynamic Decision Boundary)

* **목적**: 추론과 학습이 동시 진행될 때 상태 공간 내 결정 경계의 유동적 변화 및 수렴 여부를 검증한다.
* **입력**: 매 틱의 전체 뉴런 상태 벡터 $V(t)$ 또는 $A(t)$, 출력 뉴런의 활성화 상태.
  * 주의: Input Neuron은 $V$를 갖지 않으므로 $V(t)$를 투영 입력으로 사용할 경우 Input Neuron 차원은 0으로 채우거나 명시적으로 제외한다. 세부 처리는 구현 시 확정.
* **출력**:
  * `pca_projection_t{tick}.png`: PCA 2차원 투영 산점도 (색상=출력 뉴런 상태)
  * `tsne_projection_t{tick}.png`: t-SNE 2차원 투영 산점도
  * `separation_metrics.csv`: `tick, class_separation_score, cluster_count`
  * (선택) `pca_evolution.gif`: 시간에 따른 투영 변화 애니메이션

### 6.4 틱 성능 프로파일링 (Tick Profiling)

* **목적**: 논리적 틱과 wall-clock 시간의 관계를 모니터링하여, 향후 실시간 스케줄러 도입 여부를 판단하는 근거를 축적한다.
* **입력**: 시뮬레이터 내부의 시간 측정값 (`time.perf_counter`).
* **출력**:
  * `tick_timing.csv`: `tick, step1_ms, step2_ms, step3_ms, step4_ms, total_ms`
  * `tick_duration.png`: 틱별 총 소요 시간 시계열 그래프
  * `timing_summary.json`: 전체 실행의 평균·p50·p99·최댓값·`ticks_per_second`

### 6.5 In-Degree 분포 로깅 (In-Degree Monitoring)

* **목적**: 특정 뉴런으로의 fan-in 집중도를 관찰하고, in-degree 상한 도입이 필요한지 판단할 근거를 축적한다. 현재는 관찰만 수행한다.
* **입력**: 각 엣지의 `target` ID.
* **출력**:
  * `in_degree_hist_t{tick}.png`: in-degree 히스토그램
  * `in_degree_timeseries.csv`: `tick, neuron_id, in_degree`
  * `in_degree_summary.csv`: `tick, max, p99, mean`
  * `in_degree_warnings.log`: 사전 정의 임계(예: `N/2`)를 초과하는 뉴런 발생 시 경고

### 6.6 가중치 분포 및 Co-fire 로깅 (Weight Distribution & Co-fire Tracking)

* **목적**: 1.4의 "가중치 비포화성" 목표를 검증하고, co-activation이 잦은 시냅스 쌍의 $W_e$ 궤적을 관찰하여 자연 감쇄와 강화의 균형을 확인한다.
* **입력**: 매 틱의 엣지별 $W_e$, 그리고 co-fire 지표 $A_j(t-1) \cdot A_i(t)$.
* **출력**:
  * `weight_hist_t{tick}.png`: 전체 시냅스 가중치 히스토그램
  * `weight_boundary_ratio.csv`: `tick, ratio_W_le_0.01, ratio_W_ge_1.99` (경계 포화율)
  * `top_cofire_trajectories.png`: 상위 K개 co-fire 쌍의 $W_e$ 시계열
  * `top_cofire_data.csv`: `tick, edge_id, source, target, W, cofire_count`
  * `weight_stats.csv`: 뉴런/시냅스별 가중치 평균·분산

### 6.7 Self-loop 관찰 지표 (Self-loop Monitoring)

* **목적**: 자기 연결 시냅스가 runaway tonic firing을 유발하는지 감시한다.
* **입력**: self-loop 엣지의 $W_e$, 해당 뉴런의 $A_i(t)$.
* **출력**:
  * `selfloop_weights.csv`: `tick, neuron_id, W_selfloop`
  * `selfloop_firing.png`: self-loop 보유 뉴런의 발화 패턴 시계열
  * `selfloop_warnings.log`: 예 - 특정 뉴런이 100틱 연속 발화 시 경고 기록

---

### 6.8 출력물 디렉터리 구조

```
outputs/
├── activation/ # 6.1
├── graph/ # 6.2
├── projection/ # 6.3
├── timing/ # 6.4
├── in_degree/ # 6.5
├── weight/ # 6.6
├── selfloop/ # 6.7
└── run_meta.json # seed, config, 시작/종료 시각, 총 틱 수
```

모든 분석 모듈은 `simulator`로부터 독립적으로 실행 가능해야 하며, `run_meta.json`에 기록된 seed와 config만으로 동일 결과를 재생성할 수 있어야 한다.

---

## 7. 구현 구조 및 계획 (Implementation Structure & Plan)

본 장은 앞선 1~6장의 명세를 실제 코드로 옮기기 위한 모듈 분해, 의존성, 구현 순서, 테스트 전략을 정의한다.

### 7.1 사용 패키지 (Dependencies)

| 구분 | 패키지 | 용도 |
| :--- | :--- | :--- |
| Core | `numpy` | 벡터화 연산, 상태 배열 관리 |
| Core | `dataclasses` (표준) | 설정·뉴런·엣지 자료구조 정의 |
| Core | `random` (표준) | Seeded RNG (재현성). 스트림 분리를 위해 `random.Random(seed)` 인스턴스를 다수 사용 |
| Analysis | `networkx` | 그래프 구조 분석 및 시각화 (학습 루프에는 미사용) |
| Analysis | `matplotlib`, `seaborn` | 히트맵, 히스토그램, 시계열 시각화 |
| Analysis | `scikit-learn` | PCA, t-SNE |
| Test | `pytest` | 단위·통합 테스트 |
| Test | `hypothesis` | Property-based 테스트 (불변식 검증) |
| Runtime | `time.perf_counter` (표준) | 틱 프로파일링 |
| Runtime | `json`, `csv` (표준) | 로그·메타데이터 저장 |

* **PyTorch 미도입**: 중복 엣지·가변 그래프 구조는 순수 numpy + Python 자료구조가 더 자연스럽다. 배치 학습이 필요해지면 재검토한다.
* **RNG 스트림**: 2.3에 정의된 `init_rng`, `input_rng`, `gen_rng`를 각각 `random.Random(seed_i)`로 생성한다. seed는 상위 seed에서 파생하거나 개별 지정할 수 있으며, 어느 방식이든 최종 seed 묶음을 `run_meta.json`에 기록한다.

### 7.2 모듈 구조 (Directory Layout)

```
artificial_neuron/
├── config.py # 모든 하이퍼파라미터 dataclass
├── neuron.py # Neuron, Edge 데이터 클래스
├── network.py # Network (뉴런 컬렉션, 엣지 관리, in/out 인접 리스트, edge_id 발급)
├── input_provider.py # External Signal Provider 인터페이스 + 기본 순환 공급자
├── dynamics.py # Step 1, 2, 4: forward pass
├── plasticity.py # Step 3: Hebbian, prune, generate
├── simulator.py # tick loop + profiling + 로깅
├── analysis/
│ ├── activation.py # 6.1
│ ├── graph.py # 6.2
│ ├── projection.py # 6.3
│ ├── timing.py # 6.4
│ ├── in_degree.py # 6.5
│ ├── weight.py # 6.6
│ └── selfloop.py # 6.7
├── io/
│ └── checkpoint.py # 상태 저장/로드 (재현성)
├── tests/
│ ├── test_neuron.py
│ ├── test_network.py
│ ├── test_input_provider.py
│ ├── test_dynamics.py
│ ├── test_plasticity.py
│ ├── test_simulator.py
│ └── test_invariants.py
└── main.py # CLI 진입점
```

### 7.3 설계 원칙 (Design Principles)

1. **Network는 그래프의 유일한 소유자**: 뉴런·엣지의 생성·삭제·조회는 반드시 `Network`를 통해서만 수행한다. `Neuron`은 자신의 인접 리스트를 직접 수정하지 않는다. 이 원칙이 재현성과 디버깅 가능성을 보장한다.
2. **엣지는 명시적 ID를 가진다**: 중복 엣지가 허용되므로 `(source, target)`을 key로 쓸 수 없다. `Edge(edge_id, source, target, weight)` 형태로 관리하며, `edge_id`는 Network가 단조 증가 카운터로 발급한다. 소멸한 edge_id는 재사용하지 않는다 (로그·체크포인트 일관성 보장).
3. **in-edge / out-edge 일관성**: Network는 각 뉴런에 대해 `out_edges`와 `in_edges`를 모두 유지한다. 엣지 생성·소멸 시 양쪽 인덱스를 원자적으로 갱신한다. `dynamics`/`plasticity`는 이 인덱스를 통해 접근하며, 직접 리스트를 조작하지 않는다.
4. **단일 목적별 Seeded RNG**: 초기화, 입력(확률적 모드), 시냅스 생성은 각각 `init_rng`, `input_rng`, `gen_rng`를 통과한다. RNG 스트림과 호출 순서는 2.3에 고정되어 있으며, 임의 변경은 재현성 파괴로 간주한다.
5. **Simulator는 상태를 갖지 않는 순수 함수 조합**: 각 Step은 `step_n(network, t, ctx) -> ...` 형태로 분리한다. `network`를 in-place로 mutate하는 함수라 하더라도, 각 Step은 이전 Step의 확정 상태만을 입력으로 받는다. Step 간 상태 전달은 명시적 인자로만 이루어진다.
6. **분석은 시뮬레이션과 완전 분리**: `analysis/` 모듈은 저장된 로그·스냅샷만 읽어 동작한다. 시뮬레이터 실행 없이도 재분석 가능해야 한다.
7. **Config 일원화**: 모든 하이퍼파라미터는 `config.py`의 dataclass 하나에서 관리한다. 코드 곳곳에 매직 넘버를 두지 않는다.
8. **입력 공급자 추상화**: External Signal Provider는 `sample(t) -> tuple[int, ...]` 인터페이스를 구현한다. 기본 순환 공급자 외의 구현을 실험할 때 시뮬레이터 본체를 수정하지 않아도 되도록 한다.

### 7.4 구현 순서 (Implementation Order)

의존성 역순으로, 각 단계마다 테스트를 통과한 뒤 다음 단계로 진행한다.

| 단계 | 대상 | 테스트 |
| :--- | :--- | :--- |
| 0 | Tick 인덱스, 초기 `A(-1)`, RNG 스트림/호출 순서, ID 매핑 확정 | 본 문서 반영 확인 |
| 1 | `config.py`, `neuron.py`, `network.py` | 자료구조 단위 테스트 (edge_id 발급, in/out 일관성) |
| 2 | `input_provider.py` (기본 순환 공급자) | 패턴 순환 주기·경계값 테스트 |
| 3 | `dynamics.py` (Step 1, 2, 4) | 수식 단위 테스트 + golden test |
| 4 | `simulator.py` 최소 루프 (Step 3 미포함) | bootstrap 실측 (T=1000 로그) |
| 5 | `plasticity.py` (Step 3: Hebbian → prune → generate 순서) | 수식 단위 테스트 + 불변식 테스트 |
| 6 | `simulator.py` 완전 루프 + 프로파일링 | 통합 테스트 |
| 7 | `analysis/` 7종 | 스모크 테스트 (렌더링 확인) |
| 8 | `io/checkpoint.py` | 저장·로드 왕복 테스트 (RNG 상태 포함) |
| 9 | 종합 테스트 (장기 안정성) | 아래 7.5 참조 |

* **4단계의 중요성**: Step 3 없이 forward pass만으로 T=1000을 돌려 발화율을 측정한다. 이 결과가 1.4의 설계 목표에 부합하는지 확인하고, 필요 시 파라미터를 조정한 뒤에 Step 3를 붙인다.

### 7.5 테스트 전략 (Testing Strategy)

**(1) 수식 단위 테스트**
- `V_i(t)` 업데이트: 손으로 계산한 3-뉴런 케이스와 완전 일치. 입력 뉴런이 pre인 경우 같은 틱 $A_j(t)$, 비입력 pre는 $A_j(t-1)$을 사용하는지 별도 케이스로 검증
- `A_i(t)`: 경계값 $\theta - \epsilon$, $\theta$, $\theta + \epsilon$
- `W_e(t)` 업데이트: $(A_j, A_i)$ 4가지 조합 전수 검사, $W_{\max}$ 캡, 0 클립
- Pruning: $W_{\min}$ 경계
- Generation: stagger 조건, out-degree 캡(자기·중복 포함), Input 배제, 후보 집합 uniform 샘플링

**(2) Property-based 테스트 (불변식)**
- `0 ≤ W_e ≤ W_max` 항상 성립
- `len(out_edges(i)) ≤ 5` (self-loop, 중복 포함)
- Input neuron의 in-edge 개수 = 0
- E/I 비율 보존 (16 / 4)
- `V_i ≥ 0` (비입력 뉴런)
- `edge_id` 유일성, 소멸 후 재사용 금지
- `in_edges`와 `out_edges`의 대칭성: $e \in \text{out}(u) \iff e \in \text{in}(v)$ where $e = (u \to v)$
- Seed 고정 시 상태 완전 재현 (RNG 스트림 상태 포함)

**(3) Golden Test**
- N=5, T=20틱, seed 고정 → 기대 V, A, W 배열을 스냅샷으로 저장. 이후 변경 시 회귀 감지.
- 입력 패턴 경계(t=199→200, t=799→800 등)를 포함하도록 케이스 구성.

**(4) 통합 테스트**
- N=20, T=1000틱 정상 실행
- 각 Step이 순서대로 호출되는지
- Step 3 내부 순서가 Hebbian → prune → generate인지
- 로그 파일이 지정된 스키마로 생성되는지

**(5) 장기 안정성 테스트**
- N=20 또는 그 이상, T=10,000틱
- 발화율, in/out-degree 분포, W 분포 로깅
- 발산 / 전면 침묵 / 전면 포화 여부 자동 판정
- 1.4의 설계 목표 만족 여부 리포트

**(6) 분석 모듈 스모크 테스트**
- 저장된 로그로 각 `analysis/` 모듈이 예외 없이 실행되고 출력 파일을 생성하는지 확인 (수치 검증은 별도)

### 7.6 종합 테스트 및 결과 판정 기준

종합 테스트는 단순 통과/실패를 넘어, 다음 지표를 **참고 리포트**로 산출한다. 본 절의 수치는 모델의 건강도를 관찰하기 위한 참고 범위이며, 명시적 통과/실패 임계값이 아니다. 판정 근거는 지표의 절대값보다 **시간에 따른 변화 양상**(예: 침묵 수렴 여부, 포화 경향)에 둔다.

**발화율 지표의 그룹 구분 (중요)**:
- `input` 발화율은 외생 신호가 결정하므로 모델 건강도 판정 대상이 아니다.
- `internal`, `output` 발화율이 내부 동역학의 결과이며, 아래 참고 범위는 이 그룹을 기준으로 한다.
- `total`은 위 세 그룹의 가중 평균으로, 참고용으로만 기록한다.

| 지표 | 참고 범위 | 비고 |
| :--- | :--- | :--- |
| internal + output 평균 발화율 | ≥ 0.01 | 하한 1%는 "완전 침묵 회피"의 넉넉한 참고선. 판정 기준이 아니다. |
| internal + output 윈도우 발화율 | (관찰) | 윈도우별 시계열로 침묵 수렴 여부 확인. 별도 임계값 없음. |
| input 평균 발화율 | (참고만) | 외생 입력 패턴에 종속. 판정에 사용하지 않음. |
| W 경계 포화율 | `W ≤ 0.01` 또는 `W ≥ 1.99` 비율 < 20% | η_u, η_d, W_max 조정 여부 판단 |
| In-degree 최댓값 | `N/2` 미만 | 상한 도입 검토 |
| Self-loop runaway | 100틱 연속 발화 뉴런 0개 | self-loop 정책 재검토 |
| E/I 발화율 | 억제성 뉴런의 발화율이 0에 수렴하지 않음 | 억제 시냅스 정책 재검토 |

이 리포트를 기반으로 파라미터 조정 → 재실행 → 재평가 사이클을 반복한다.

**하한 1%의 의미**: Config의 파라미터(θ, W_init, γ 등)는 본 조항으로 인해 자동 변경되지 않는다. 하한 미달이 관찰되면 Architect가 판단하여 조정 여부를 결정한다.

### 7.7 재현성 및 로깅 (Reproducibility)

* **Seed 고정**: `run_meta.json`에 사용된 seed 묶음(`init_seed`, `input_seed`, `gen_seed`), config 전체, 시작·종료 시각, 총 틱 수를 기록한다.
* **RNG 상태 저장**: 각 RNG 스트림의 내부 상태(`random.getstate()` 결과)를 체크포인트에 저장한다. 재개 시 이 상태에서 이어서 실행하면 완전 재현된다.
* **Config 스냅샷**: 실행 시점의 `config` 객체를 JSON으로 저장하여, 이후 코드 변경과 무관하게 결과를 재생성할 수 있도록 한다.
* **체크포인트 저장 항목**:
  * `tick` (현재 틱 인덱스)
  * 각 뉴런의 속성 ($V_i$, $A_i$, $S_i$, ID)
  * 전체 엣지 리스트 (`edge_id, source, target, weight`)
  * `edge_id_counter` (다음 발급 번호)
  * RNG 스트림 상태 묶음
  * config 스냅샷
* **로그 스키마**: 모든 CSV 로그는 첫 행에 헤더를 포함하고, 컬럼 순서를 고정한다.