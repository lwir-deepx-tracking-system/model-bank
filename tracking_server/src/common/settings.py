"""실행 중에 바뀌는 설정 네 가지: 온도 거름 최소·최대(WARM_MIN, WARM_MAX), 가장자리 출생 점수(EDGE_BIRTH), 가장자리 폭(EDGE_PX).

시작 옵션과 화면의 설정 바꾸기가 이 값을 바꾼다. 다른 파일에서는 반드시 settings.WARM_MIN 처럼 이 모듈을 거쳐 읽고 쓴다.
(from ... import 로 이름만 가져가면 나중에 바뀐 값이 보이지 않는다.)"""


# v24f (11-10, user approved both rules): WARM_MIN = raw 16-bit 95th percentile a person box must reach (0 = off). Measured 11-10 on this camera (AGC on):
# person boxes p95 >= 47,749 in the room, >= 60,170 close-up B, cut-off slivers down to 40,387; ceiling lights (v9 false tracks) <= 43,397; warm wall object <= 49,155.
# EDGE_BIRTH = score a detection touching a frame edge (within EDGE_PX) needs to START a track (existing tracks keep matching at lower scores); 0 = off.
WARM_MIN, EDGE_BIRTH, EDGE_PX = 0, 0.0, 4.0
WARM_MAX = 0   # v33: upper limit of the warmth range (0 = none)
