"""Desktop entry point: start the board over wired SSH, then open the PC app."""
from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

from app import App
from link import Link
from remote_start import PasswordRequired, StartupError, SwitchRequired, start


CONFIG = Path.home() / '.config' / 'modelbank' / 'direct_lan.json'


def load_config():
    try:
        data = json.loads(CONFIG.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_config(data):
    CONFIG.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    handle = os.open(CONFIG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(handle, 'w') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


class DesktopStart:
    def __init__(self, root, settings):
        self.root = root
        self.settings = settings
        self.password = None
        self.allow_web_switch = False
        self.events = queue.Queue()
        self.answers = queue.Queue(maxsize=1)
        self.busy = False
        root.title('ModelBank · 직결 LAN 시작')
        root.geometry('480x210')
        box = ttk.Frame(root, padding=20)
        box.pack(fill='both', expand=True)
        self.box = box
        ttk.Label(box, text='ModelBank 직결 LAN', font=('sans', 16, 'bold')).pack(anchor='w')
        self.message = tk.StringVar(value='Orange Pi 연결 상태를 확인합니다…')
        ttk.Label(box, textvariable=self.message, wraplength=430).pack(anchor='w', pady=15)
        self.retry = ttk.Button(box, text='다시 시도', command=self.launch)
        self.retry.pack(anchor='e')
        self.retry.state(['disabled'])
        root.after(100, self.pump)
        self.launch()

    def launch(self):
        if self.busy:
            return
        self.busy = True
        self.allow_web_switch = False
        self.retry.state(['disabled'])
        self.message.set('Orange Pi 연결 상태를 확인합니다…')
        thread = threading.Thread(target=self.work, name='modelbank-remote-start', daemon=True)
        thread.start()

    def work(self):
        try:
            while True:
                try:
                    start(self.settings['board'], self.settings['port'], self.settings['ssh_user'],
                          password=self.password, remote_root=self.settings.get('remote_root', ''),
                          progress=lambda value: self.events.put(('progress', value)),
                          allow_web_switch=self.allow_web_switch)
                    self.events.put(('ready', None))
                    return
                except SwitchRequired as exc:
                    self.events.put(('switch', str(exc)))
                    self.allow_web_switch = self.answers.get()
                    if not self.allow_web_switch:
                        self.events.put(('error', '기존 웹 실행을 유지했습니다'))
                        return
                except PasswordRequired:
                    self.events.put(('password', None))
                    self.password = self.answers.get()
                    if self.password is None:
                        self.events.put(('error', 'SSH 비밀번호 입력을 취소했습니다'))
                        return
        except (StartupError, OSError, ValueError) as exc:
            self.events.put(('error', str(exc)))

    def pump(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'progress':
                    self.message.set(value)
                elif kind == 'password':
                    answer = simpledialog.askstring('Orange Pi SSH',
                        'Orange Pi SSH 비밀번호를 입력하세요. 비밀번호는 저장하지 않습니다.',
                        show='*', parent=self.root)
                    self.answers.put(answer)
                elif kind == 'switch':
                    self.answers.put(messagebox.askyesno('ModelBank 직결 LAN', value, parent=self.root))
                elif kind == 'error':
                    self.password = None
                    self.busy = False
                    self.message.set(value)
                    self.retry.state(['!disabled'])
                    messagebox.showerror('ModelBank 직결 LAN', value, parent=self.root)
                elif kind == 'ready':
                    self.box.destroy()
                    App(self.root, Link(self.settings['board'], self.settings['port']))
                    return
        except queue.Empty:
            pass
        self.root.after(100, self.pump)


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description='Start ModelBank on the cabled Orange Pi, then open the PC app')
    parser.add_argument('--board', default=config.get('board', '192.168.77.2'))
    parser.add_argument('--port', type=int, default=config.get('port', 8765))
    parser.add_argument('--ssh-user', default=config.get('ssh_user', ''))
    parser.add_argument('--remote-root', default=config.get('remote_root', ''))
    args = parser.parse_args()
    root = tk.Tk()
    if not args.ssh_user:
        user = simpledialog.askstring('Orange Pi SSH', 'Orange Pi의 SSH 사용자 이름을 입력하세요.', parent=root)
        if not user:
            root.destroy()
            return
        args.ssh_user = user.strip()
    settings = dict(board=args.board, port=args.port, ssh_user=args.ssh_user,
                    remote_root=args.remote_root)
    if settings != config:
        save_config(settings)
    DesktopStart(root, settings)
    root.mainloop()


if __name__ == '__main__':
    main()
