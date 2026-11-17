// No hardware: exercise the actual executor and vendor command encoder.
#include "control/gimbal_command_executor.hpp"
#include <atomic>
#include <cassert>
#include <iostream>
#include <thread>

struct Device : IGimbalDevice {
    std::atomic<int> stops{0}, homes{0};
    bool fail_home = false;
    void move(int, int) override {}
    void stop() override { ++stops; }
    void home() override {
        if (fail_home) throw std::runtime_error("home write failed");
        ++homes;
    }
};

int main() {
    ControlConfig config;
    config.enabled = true;
    config.heartbeat_timeout_ms = 100;
    RuntimeControl runtime;
    auto device = std::make_unique<Device>();
    auto* recorded = device.get();
    GimbalCommandExecutor executor(config, std::move(device), runtime);
    assert(executor.home() == ControlResult::NotReady && recorded->homes == 0);
    executor.connected();
    executor.move(1, 0);
    const auto before = recorded->stops.load();
    assert(executor.home() == ControlResult::Ok);
    assert(recorded->stops == before + 1 && recorded->homes == 1);
    auto state = executor.snapshot();
    assert(!state.automatic && state.selected_id == 0 && state.moving);
    executor.halt();
    assert(!executor.snapshot().moving && recorded->stops == before + 2);
    auto frame = std::make_shared<FrameContext>();
    frame->metadata.frame_id = 1;
    frame->metadata.captured_at = std::chrono::steady_clock::now();
    frame->image = cv::Mat(480, 640, CV_16UC1);
    auto result = std::make_shared<TrackingResult>();
    result->frame = frame;
    Track person;
    person.track_id = 7; person.x = 300; person.y = 200;
    person.width = 40; person.height = 80; person.confidence = 0.9f;
    result->tracks.push_back(person);
    executor.publish(result, true);
    executor.snapshot(); // synchronize with the latest-result mailbox
    assert(executor.select(7) == ControlResult::Ok);
    executor.ping();
    assert(executor.home() == ControlResult::Ok);
    state = executor.snapshot();
    assert(!state.automatic && !state.selected_id && state.moving);
    std::this_thread::sleep_for(std::chrono::milliseconds(160));
    assert(!executor.snapshot().moving); // heartbeat still stops an absolute move
    executor.connected();
    recorded->fail_home = true;
    assert(executor.home() == ControlResult::NotReady);
    state = executor.snapshot();
    assert(state.error && !state.moving && !state.automatic);
    const auto json = WaveshareCommands::home(config.gimbal);
    assert(json.find("\"T\":133,\"X\":0,\"Y\":0,") != std::string::npos);
    std::cout << "home executor: disconnected refusal, STOP before home, cancellation state, halt, heartbeat, write failure, zero-angle command passed\n";
}
