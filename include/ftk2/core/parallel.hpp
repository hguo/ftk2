#pragma once

#include <algorithm>
#include <vector>
#include <thread>
#include <mutex>

#if defined(FTK_HAVE_OPENMP)
#include <omp.h>
#endif

namespace ftk2 {

template <typename IndexType, typename Function>
void parallel_for(IndexType start, IndexType end, Function func) {
#if defined(FTK_HAVE_OPENMP)
    #pragma omp parallel for
    for (IndexType i = start; i < end; ++i) {
        func(i);
    }
#else
    // Fallback to std::thread (C++11)
    unsigned int n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 4; // Default if detection fails
    
    IndexType count = end - start;
    if (count <= 0) return;
    
    // If work is small, avoid overhead
    if (count < (IndexType)n_threads) {
        for (IndexType i = start; i < end; ++i) func(i);
        return;
    }

    std::vector<std::thread> threads;
    IndexType chunk_size = count / n_threads;
    IndexType remainder = count % n_threads;
    
    IndexType current_start = start;
    for (unsigned int t = 0; t < n_threads; ++t) {
        IndexType this_chunk = chunk_size + (t < remainder ? 1 : 0);
        IndexType current_end = current_start + this_chunk;
        
        threads.emplace_back([current_start, current_end, &func]() {
            for (IndexType i = current_start; i < current_end; ++i) {
                func(i);
            }
        });
        current_start = current_end;
    }

    for (auto& t : threads) {
        if (t.joinable()) t.join();
    }
#endif
}

} // namespace ftk2
