/*
Copyright (c) Ishwar R. Kulkarni
All rights reserved.

This file is part of TextMining Project by 
Ishwar Kulkarni , see https://github.com/IshwarKulkarni/NeuralNetworks

If you so desire, you can copy, redistribute and/or modify this source 
along with  rest of the project. However any copy/redistribution, 
including but not limited to compilation to binaries, must carry 
this header in its entirety. A note must be made about the origin
of your copy.

TextMining is being distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
FITNESS FOR A PARTICULAR PURPOSE.

*/ 

#ifndef __UTILS_INCLUDED__
#define __UTILS_INCLUDED__

#include <chrono>
#include <cstring>
#include <numeric>
#include <thread>



static const double Eps = std::numeric_limits<double>::min();

namespace Utils {

    inline double TimeSince(std::chrono::steady_clock::time_point since)
    {
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> duration =
            std::chrono::duration_cast<std::chrono::duration<double>>(now - since);
        return duration.count();
    }

    // Launches MaxThreads threads and waits for all to finish, before launching next
    // Inefficient if tasks times are very different && pool size is small.
    struct ThreadPool
    {
    private:
        size_t MaxThreads;
        std::vector<std::thread> Threads;

    public:

        ThreadPool(size_t N = -1) : MaxThreads(N == size_t(-1) ? std::thread::hardware_concurrency() - 1 : N) { };

        size_t GetPoolSize() const { return MaxThreads; }

        void ResetPoolSize(size_t N = -1) {
            JoinAll();
            MaxThreads = (N == size_t(-1) ? std::thread::hardware_concurrency() - 1 : N);
        }

        template<typename Func, typename ... Args>
        void Launch(Func&& func, Args&&... args)
        {
            if (Threads.size() == MaxThreads) JoinAll();

            Threads.push_back(std::thread(std::forward<Func>(func), std::forward<Args>(args)...));
        }

        void JoinAll()
        {
            for (auto& t : Threads)
                t.join();

            Threads.clear();
        }

        ~ThreadPool() { JoinAll(); }
    };
}
#endif
