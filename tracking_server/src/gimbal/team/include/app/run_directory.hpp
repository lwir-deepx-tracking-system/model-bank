#pragma once

#include <filesystem>

// 같은 초에 여러 번 실행해도 기존 결과를 보존하는 고유 실행 폴더를 만든다.
std::filesystem::path create_run_directory(const std::filesystem::path &root);
