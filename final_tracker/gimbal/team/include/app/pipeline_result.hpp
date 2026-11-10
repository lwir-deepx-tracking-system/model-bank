#pragma once

#include <cstdint>

// 콘솔과 통신 계층에서 공통으로 사용하는 파이프라인 요청 결과.
enum class PipelineResult : std::uint8_t
{
    Ok             = 0,
    InvalidId      = 1,
    Busy           = 2,
    CreationFailed = 3,
    Restored       = 4,
    RestoreFailed  = 5,
    NotReady       = 6
};
