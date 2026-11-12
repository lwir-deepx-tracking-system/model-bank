#!/bin/bash
# 장치(DX-M1, Orange Pi 5 Plus)에서 한 번 돌리는 준비 스크립트. 어느 폴더에서 불러도 된다:  bash tracking_server/scripts/setup.sh
# 하는 일 (모두 tracking_server/ 안에서):
#   1. src/detect/post.cpp -> src/detect/post.so            검출기 출력을 푸는 빠른 계산 (native_post.py 가 부른다)
#   2. models/links/<내부 이름>.dxnn 링크 만들기             서버는 모델을 내부 이름으로 찾는다. 내부 이름 -> 파일 표는 models/MANIFEST.json 의 "_links"
#   3. models/MANIFEST.json 의 확인값(sha256)으로 모델 파일 확인   내부 이름으로 연결된 것과, models/detect/*/ · models/reid/ 의 모든 DXNN ("_files")
#   4. src/camera/cam_pipe.cpp -> src/camera/cam_pipe        카메라 프로그램 (i3system 카메라 SDK 가 있어야 한다)
#   5. src/gimbal/gimbal_bridge_v25.cpp + src/gimbal/team/ -> src/gimbal/gimbal_bridge_v25    짐벌 + PC GUI 프로그램 (OpenCV, GStreamer 가 있어야 한다)
# 내부 이름이 없는 검출 모델은 링크가 필요 없다: 서버가 models/detect/ 와 그 바로 아래 폴더에서 스스로 찾아 화면의 모델 목록에 올린다.
set -eu
cd "$(dirname "$0")/.."

g++ -O3 -shared -fPIC -o src/detect/post.so src/detect/post.cpp
echo "post.so built"

if ! bash scripts/build_native_trackers.sh; then
  echo "Optional C++ tracker build failed; check the build log. Unsupported presets use Python." >&2
fi

mkdir -p models/links
python3 - <<'EOF'
import hashlib, json, os
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


man = json.loads(Path('models/MANIFEST.json').read_text(encoding='utf-8'))
links, files = man.get('_links', {}), man.get('_files', {})
bad = []
# links: models/links/<name>.dxnn -> ../<file>
for name, rel in links.items():
    link = Path('models/links') / (name + '.dxnn')
    if not (Path('models') / rel).is_file():
        print(name, 'MISSING', rel)
        bad.append(name)
        continue
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(os.path.join('..', rel), link)
for old in sorted(Path('models/links').glob('*.dxnn')):
    if old.stem not in links:
        print(old.stem, 'is not in MANIFEST.json "_links": link left as it is')
# check: every internal name, then every model file
for name, want in man.items():
    if name.startswith('_') or name in bad:
        continue
    if name not in links:
        print(name, 'has no entry in "_links"')
        bad.append(name)
        continue
    got = sha(Path('models/links') / (name + '.dxnn'))
    print(name, 'OK' if got == want else 'MISMATCH', got[:12], '->', links[name])
    if got != want:
        bad.append(name)
n_ok = 0
for rel, want in files.items():
    p = Path('models') / rel
    if not p.is_file():
        print(rel, 'MISSING (not needed by a linked model: skipped)' if rel not in links.values() else 'MISSING')
        if rel in links.values():
            bad.append(rel)
        continue
    got = sha(p)
    if got != want:
        print(rel, 'MISMATCH', got[:12])
        bad.append(rel)
    else:
        n_ok += 1
print('model files checked: %d of %d OK' % (n_ok, len(files)))
for sub in ('detect', 'reid'):
    for p in sorted((Path('models') / sub).rglob('*.dxnn')):
        rel = p.relative_to('models').as_posix()
        if rel not in files:
            print(rel, 'is not in MANIFEST.json (new file: not checked)')
if bad:
    raise SystemExit('setup stopped: model files missing or different from MANIFEST.json: ' + ', '.join(bad))
EOF

# 카메라 프로그램. SDK 가 없으면 실시간 서버만 못 쓰고, 파일로 돌리는 실행은 된다.
# 셔터 설정 파일 cam_shutter.txt 를 쓰는 장치에서는 그 파일을 src/camera/ (cam_pipe 옆) 에 둔다. 파일이 없으면 cam_pipe 는 기본 셔터 설정으로 연다.
if [ -f /usr/local/include/i3system/i3system_TE.h ]; then
  g++ -O2 src/camera/cam_pipe.cpp -o src/camera/cam_pipe -D_LINUX -I/usr/local/include/i3system $(pkg-config --cflags --libs opencv4) -li3system_te_64 -li3system_usb_64 -li3system_imgproc_impl_64 -lpthread
  echo "cam_pipe built"
else
  echo "camera SDK not found (/usr/local/include/i3system): cam_pipe not built, the live server will not start"
fi

# 짐벌 + PC GUI 프로그램: src/gimbal/gimbal_bridge_v25.cpp 와 팀의 제어·GUI 코드 src/gimbal/team (그대로 가져온 사본, team/SOURCE.txt 참고).
# OpenCV core/imgproc 와 GStreamer (gstreamer-1.0, gstreamer-app-1.0) 가 있어야 한다. 없으면 --gimbal 과 --gui-host 를 못 쓰므로 scripts/run_real_gui.sh 가 시작하지 않는다.
if pkg-config --exists opencv4 gstreamer-1.0 gstreamer-app-1.0; then
  G=src/gimbal
  g++ -O2 -std=c++17 -pthread -I$G/team/include $(pkg-config --cflags opencv4 gstreamer-1.0 gstreamer-app-1.0) $G/gimbal_bridge_v25.cpp \
    $G/team/src/control/auto_tracking_controller.cpp $G/team/src/control/gimbal_command_executor.cpp $G/team/src/control/gimbal_device.cpp $G/team/src/control/pid_controller.cpp \
    $G/team/src/control/tracking_command_handler.cpp $G/team/src/gui/gui_protocol.cpp $G/team/src/gui/gui_sender.cpp $G/team/src/gui/gui_receiver.cpp $G/team/src/gui/gui_command_handler.cpp \
    $G/team/src/common/logger.cpp $(pkg-config --libs gstreamer-1.0 gstreamer-app-1.0) -lopencv_core -lopencv_imgproc -o $G/gimbal_bridge_v25.next
  mv -f "$G/gimbal_bridge_v25.next" "$G/gimbal_bridge_v25"
  echo "gimbal_bridge_v25 built"
else
  echo "OpenCV (opencv4) or GStreamer not found: gimbal_bridge_v25 not built, --gimbal and --gui-host will not start"
fi
echo "setup done: post.so built, models linked and checked"
