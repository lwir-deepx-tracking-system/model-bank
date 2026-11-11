#pragma once

#include <csignal>

// signal handler는 async-signal-safe한 플래그만 변경하고 I/O·정리는 호출 애플리케이션에 맡긴다.
namespace app_shutdown
{
// 한 프로세스의 최상위 실행 범위 하나가 소유하며 작업 스레드 시작 전에 생성한다.
class SignalScope
{
    using Handler         = void (*)(int);
    const int signals_[3] = {SIGINT, SIGTERM, SIGQUIT};
    Handler   previous_[3]{};
    unsigned  installed_ = 0;

  public:
    SignalScope();
    ~SignalScope();
    SignalScope(const SignalScope &)            = delete;
    SignalScope &operator=(const SignalScope &) = delete;
};
inline volatile std::sig_atomic_t requested       = 0;
inline volatile std::sig_atomic_t discard_results = 0;

inline void reset()
{
    requested       = 0;
    discard_results = 0;
}
inline void handle_signal(int signal)
{
    if(signal == SIGQUIT) discard_results = 1;
    requested = 1; // 이후 SIGINT/SIGTERM을 받아도 SIGQUIT의 폐기 요청은 유지한다.
}
inline void request_stop() { requested = 1; } // 콘솔 quit는 SIGINT와 같은 저장 종료다.
} // namespace app_shutdown
