"""실시간 추적 서버의 코드 묶음. 일별 하위 폴더: detect(검출), track(추적), reid(재식별), camera(카메라), gimbal(짐벌), web(웹 화면), common(공통).

이 파일은 묶음의 어느 부분보다도 먼저 실행된다. 계산 라이브러리(numpy 등)를 불러오기 전에
스레드 수 환경 변수를 정해야 해서 그 한 줄을 여기에 둔다."""
import os


os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
