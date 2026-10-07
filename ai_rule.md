# AI Collaboration Guidelines (26-10-07)

본 문서는 Architect(User)와 Engine(AI)가 Develop 과정에서 전체적인 프로세스와 각 프로세스에서 각각의 역할 및 진행해야 할 작업들에 대한 규칙을 정의하는 문서이다. 따라서 본 문서 'ai_rule.md'의 내용은 프로젝트를 진행할 때 가장 기본이 되는 규칙이다. 이 문서의 수정은 개발 과정에서 일어나지 않으며, 별도의 작업으로 진행된다.

## step 1. 초기 버전 또는 다음 버전에서의 develop 방향성 공유 및 통합된 방향성 도출

Architect와 Engine이 생각한 각각의 방향성을 서로 공유한다. 그리고 이번 버전의 개발 방향성을 하나로 통합해 도출한다. 이 단계가 끝나면 업데이트 버전 + 업데이트 방향성이 나오게 된다.

## step 2. 'algorithm_spec.md' 문서를 방향성에 맞게 생성 또는 수정.

'algorithm_spec.md' 문서는 Architect 주도 하에 결정된 설계 내용을 작성하는 문서이다. 이 문서는 Architect가 사용하는 언어인 자연어, 수학으로 그 알고리즘 명세가 표현되어 있다. 논의된 방향성을 통해 'algorithm_spec.md' 문서를 생성 또는 수정한다. 먼저 Architect가 세부 수정사항을 제시한다면 이를 Engine이 문서에 반영하는 것으로 작업이 시작된다. 그 이후에는 Architect의 feedback - Engine의 반영 루프를 통해 algorithm_spec.md 문서가 다음 개발 단계로 넘어가도 되는지 Architect가 판단한다. Architect의 판단이 필요한 feedback마다 이 루프를 진행하고, 통과 시 다음 단계로 넘어간다.

## Step 3. 'programming_plan.md' 문서를 'algorithm_spec.md'의 설계 내용에 맞게 생성 또는 수정.

'programming_plan.md' 문서는 Engine 주도 하에 프로젝트 repository 구조, 사용할 패키지 및 라이브러리 목록, 구현하는 언어 및 함수에 대한 사용 계획 등의 종합적인 구현 계획을 내용으로 작성하는 문서이다. 이 문서는 해당 버전의 산출물로 repository에 남긴다.
- 먼저 해당 단계가 시작될 때, Engine은 Architect에게 다음 사항을 질문한다.
    - 1. 해당 단계에서 요구하는 사항의 유무와 있다면 그게 무엇인지
    - 2. 구현을 위해 Architect가 추가로 의사결정을 해야 하는 항목
- Architect는 두 가지에 대한 답을 Engine에게 제출한다. 의사결정 결과와 추가로 요구사항이 있다면 그 요구사항을 반영하여 구현 계획을 세우고 이를 'programming_plan.md'에 반영한다.
- 첫 반영 이후 Engine은 작성한 'programming_plan.md'에 대한 self feedback을 한다. feedback에서는 다음 단계로 넘어가도 될지에 대한 판단을 하고 통과 못할 시, feedback 내용을 반영해 'programming_plan.md' 수정을 진행하고 다시 feedback을 하는 루프를 진행한다. Engine이 통과로 판단하면 해당 단계의 진행 상황과 판단 근거를 Architect에게 보고하고 승인을 요청한다. Architect가 승인한 경우에만 다음 단계로 넘어간다.

## step 4. 코드 구현 및 테스트, 'report.md' 생성 또는 업데이트

Engine은 'programming_plan.md' 문서의 내용을 따라 구현에 들어간다. 코드 수정 및 테스트 활동이 가능하다. 모든 구현 및 테스트가 마치면 해당 단계의 과정을 종합적으로 정리한 'report.md' 문서를 생성 또는 업데이트한다. 이 문서는 해당 버전의 산출물로 repository에 남긴다. 이때, 문서의 내용은 코드 자체의 설명보다는 'algorithm_spec.md' 문서에 있는 알고리즘 및 의도가 잘 반영되었는지를 위주로 구성한다. Engine이 report 작성을 마치면 Architect는 'algorithm_spec.md'와의 일치 여부 및 테스트 결과를 검토한다. Architect가 검토를 완료하고 승인한 경우에만 다음 단계로 넘어간다.

## step 5. 'README.md', 'CHANGELOG.md' 생성 또는 업데이트

이번 단계에서는 두개의 문서를 생성 또는 업데이트하는 작업을 한다.
- README.md : Engine의 주도 하에 다음 두 카테고리의 내용을 구성한다.
    - 1. project 개요, 소개
    - 2. 사용법
- CHANGELOG.md : 크게 두 카테고리 내용으로 구성된다.
    - 1. 업데이트 내용: Engine의 주도 하에 이전 버전과 바뀐 점, 개선된 점 등을 설명한 내용으로 구성한다.
    - 2. 개발 일지: Architect가 직접 수정하는 영역으로 개발 과정에서 했던 생각 및 기억나는 사건 등을 기록하는 영역이다. 그리고 이 부분은 step 1~4 과정이 진행되는 중간에 수시로 그 내용을 수정 가능하다.

## step 6. git 처리

아직 구체화한 process가 없으므로 Architect 주도 하에 stage, commit, branch 관리 및 원격 저장소 push, pr를 진행한다.