#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstdint>
#include <queue>
#include <utility>

// 여러 thread 사이에서 데이터를 안전하게 전달하기 위한 공통 Queue.
template <typename T>
class ThreadSafeQueue
{
private:
    std::queue<T> queue_;

    // 여러 Thread의 Queue 동시 접근을 방지
    std::mutex mutex_;

    // Queue가 비어 있을 때 Consumer를 대기시키고,
    // 데이터가 들어오거나 종료될 때 깨우기 위해 사용
    std::condition_variable cond_;

    // Queue 종료 상태
    bool closed_;
    std::uint64_t interrupt_generation_ = 0;
    
public:
    // 중단 가능한 pop의 결과를 데이터 수신·상태 전환·큐 종료로 구분한다.
    enum class PopResult { Item, Interrupted, Closed };
    // Mutex, Condition Variable 초기화
    ThreadSafeQueue()
        : closed_(false)
    {
    }

    // Mutex, Condition Variable 자원 해제
    ~ThreadSafeQueue()
    {
    }

    // 동기화 객체를 포함하므로 복사 금지
    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

    // 데이터를 Queue에 삽입하고 대기 중인 Consumer 하나를 깨운다.
    // Queue가 이미 닫힌 경우 false를 반환한다.
    bool push(T data)
    {
        {
            std::lock_guard lock(mutex_);
            if(closed_)
            {
                return false;
            }

            queue_.push(std::move(data));
        }

        cond_.notify_one();
        return true;
    }

    // 최신 상태가 중요한 GUI 경로에서 사용한다.
    // max_size를 넘으면 가장 오래된 항목을 제거하고 새 항목을 넣는다.
    bool push_latest(T data, std::size_t max_size)
    {
        {
            std::lock_guard lock(mutex_);
            if (closed_ || max_size == 0)
            {
                return false;
            }

            while (queue_.size() >= max_size)
                queue_.pop();

            queue_.push(std::move(data));
        }

        cond_.notify_one();
        return true;
    }

    // Queue에서 데이터를 하나 꺼낸다.
    // Queue가 비어 있으면 데이터가 들어오거나 close()될 때까지 대기한다.
    // true  : 정상적으로 데이터를 꺼냄
    // false : Queue가 닫혔고 남은 데이터도 없음
    bool pop(T& data)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        while (queue_.empty() && !closed_)
        {
            cond_.wait(lock);
        }

        if (queue_.empty() && closed_)
        {
            return false;
        }

        data = std::move(queue_.front());
        queue_.pop();

        return true;
    }

    // 대기하지 않고 현재 큐의 항목 하나를 꺼낸다.
    bool try_pop(T& data)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;
        data = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    // A control transition can wake a consumer even when no new item arrives.
    PopResult pop_interruptible(T& data, std::uint64_t& observed_interrupt)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [&] { return closed_ ||
            interrupt_generation_ != observed_interrupt || !queue_.empty(); });
        if (interrupt_generation_ != observed_interrupt) {
            observed_interrupt = interrupt_generation_;
            return PopResult::Interrupted;
        }
        if (queue_.empty()) return PopResult::Closed;
        data = std::move(queue_.front());
        queue_.pop();
        return PopResult::Item;
    }

    // 항목을 추가하지 않고 pop_interruptible() 대기자를 깨운다.
    void interrupt_waiters() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++interrupt_generation_;
        }
        cond_.notify_all();
    }

    // Queue를 종료 상태로 만들고,
    // pop()에서 대기 중인 모든 Thread를 깨운다.
    void close()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        cond_.notify_all();
    }

    // 현재 Queue가 비어 있는지 확인한다.
    bool empty()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        bool result = queue_.empty();
        return result;
    }

    // 현재 Queue에 저장된 데이터 개수를 반환한다.
    std::size_t size()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t result = queue_.size();
        return result;
    }
};
