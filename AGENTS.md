# 이 저장소에서 일하는 AI 작업자(Codex, Claude 등)에게

1. **먼저 [`HANDOFF.md`](HANDOFF.md) 를 끝까지 읽습니다.** 지금 상태, 장치에 안 내려간 것, 다음 할 일, 알려진 문제, 일하는 규칙이 있습니다.
2. 처음 한 번, 이 저장소를 받은 곳에서 push 규칙 검사를 켭니다:

   ```bash
   git config core.hooksPath .githooks
   ```

3. 설명서: [`README.md`](README.md) (두 서버 소개) → [`tracking_server/README.md`](tracking_server/README.md) (추적 서버) / [`label_server/README.md`](label_server/README.md) (라벨 서버) / [`tracking_server/models/README.md`](tracking_server/models/README.md) (모델).

## 반드시 지킬 것

- **push 할 때마다 `HANDOFF.md` 를 고칩니다** (그 파일 7절 "push 할 때"). 고치지 않으면 `.githooks/pre-push` 가 push 를 막습니다. 검사를 끄거나 건너뛰지 않습니다 (`--no-verify` 금지).
- 공개 저장소입니다. 장치 주소, 키, 토큰, 계정, 개인 경로(`/home/<이름>`, `C:\Users\<이름>`), 카메라 영상, 라벨 자료를 올리지 않습니다.
- 짐벌이 움직일 수 있는 일은 사람이 짐벌 옆에 있을 때만 합니다.
- 장치 재부팅, 파일 지우기, 다른 사람 폴더는 장치 주인에게 먼저 묻습니다.
- 바꾼 뒤에는 "고치기 전과 결과가 같은지" 확인하고, 방법과 숫자를 커밋 글과 `HANDOFF.md` 에 적습니다. 확인 못 한 것은 못 했다고 적습니다.
- 사용자와는 한국어로, 쉬운 말로 이야기합니다.
