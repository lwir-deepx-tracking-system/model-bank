#pragma once

#include <iosfwd>
#include <string>
#include <vector>
#include "app/pipeline_requests.hpp"

// 테스트 콘솔: 입력을 기다리지 않고 애플리케이션 루프에서 요청·완료 결과를 처리한다.
class PipelineConsole
{
    struct Pending
    {
        std::string                        command;
        std::shared_future<PipelineResult> completion;
    };
    IPipelineRequests   &requests_;
    std::ostream        &output_;
    int                  input_fd_;
    bool                 input_open_ = true, discarding_ = false, quitting_ = false;
    std::string          line_;
    std::vector<Pending> pending_;

  public:
    PipelineConsole(IPipelineRequests &requests, std::ostream &output, int input_fd = 0);
    // 한 줄을 검증하고 공용 파이프라인 요청 인터페이스로 전달한다.
    bool execute_line(const std::string &line);
    // 입력과 완료 future를 비차단으로 확인한다. quit 입력이면 false를 반환한다.
    bool poll();
};
