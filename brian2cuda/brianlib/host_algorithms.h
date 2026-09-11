#ifndef BRIAN_HOST_ALGORITHMS_H
#define BRIAN_HOST_ALGORITHMS_H

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace brian {

template<typename Key, typename Value>
void sort_by_key(Key* keys, Value* values, size_t n)
{
    if (n <= 1)
        return;
    std::vector<std::pair<Key, Value> > zipped(n);
    for (size_t i = 0; i < n; ++i)
        zipped[i] = std::pair<Key, Value>(keys[i], values[i]);
    std::sort(zipped.begin(), zipped.end(),
              [](const std::pair<Key, Value>& a,
                 const std::pair<Key, Value>& b) {
                  return a.first < b.first;
              });
    for (size_t i = 0; i < n; ++i)
    {
        keys[i] = zipped[i].first;
        values[i] = zipped[i].second;
    }
}

template<typename Key, typename Value>
size_t unique_by_key(Key* keys, Value* values, size_t n)
{
    if (n == 0)
        return 0;
    size_t out_n = 1;
    for (size_t i = 1; i < n; ++i)
    {
        if (keys[i] != keys[out_n - 1])
        {
            keys[out_n] = keys[i];
            values[out_n] = values[i];
            ++out_n;
        }
    }
    return out_n;
}

}  // namespace brian

#endif
