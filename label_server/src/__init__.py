# -*- coding: utf-8 -*-
"""라벨 검수 사이트 서버의 부품 모음. 서버를 켜는 파일은 한 칸 위의 start_label_server.py 다.

  state.py             여러 부품이 함께 쓰는 고정값과, 서버가 켜져 있는 동안의 기억
  sequences.py         라벨 폴더에서 시퀀스를 찾고, 한 시퀀스의 진행 상태를 읽는다
  list_page.py         첫 화면(시퀀스 목록)의 글
  editor_page.py       편집 화면: label_editor.html 에 서버 저장 기능을 끼워 넣는다
  file_ops.py          파일을 바꾸는 일: 저장, 영상 파일 치우기, 시퀀스를 휴지통으로 보내기, 이름 바꾸기
  handler.py           HTTP 요청을 받아 주소별로 나눠 주는 부분
  auto_add.py          30초마다 auto_add_measure.py 를 돌리는 배경 작업
  auto_add_measure.py  따로 도는 프로그램 (이 패키지가 불러 쓰지 않는다). 측정 녹화를 시퀀스로 바꿔 라벨 폴더에 넣는다
"""
