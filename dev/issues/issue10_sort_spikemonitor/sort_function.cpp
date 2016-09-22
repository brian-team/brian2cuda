#include <vector>
#include <iostream>
#include <stdint.h>
#include <algorithm>
#include <typeinfo>


template<typename T>
class ArgSort
{
   private:
     	std::vector<T>* _time;
   public:
	ArgSort(
		std::vector<T>* time
		) : _time(time) {}

	bool operator() (const size_t& a, const size_t& b) const{
		return _time->at(a) < _time->at(b);
	}
};


void sort_spikemonitor(std::vector<double>& monitor_t, std::vector<int32_t>& monitor_i)
{
	// better to intialize some size_t size = monitor_t.size() only once?
	std::vector<size_t> idx(monitor_t.size());

	// initialize indices
	for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;


	// sort indices accordint to spike times
	std::sort(idx.begin(), idx.end(), ArgSort<double>(&monitor_t));

	std::vector<double> sorted_monitor_t(monitor_t.size());
	std::vector<int32_t> sorted_monitor_i(monitor_i.size());

	// fill sorted monitor vectors
	for (size_t i = 0; i != idx.size(); ++i) {

		sorted_monitor_t[i] = monitor_t[idx[i]];
		sorted_monitor_i[i] = monitor_i[idx[i]];
	}

	// copy sorted monitor into original monitor
	for (size_t i = 0; i != idx.size(); ++i) {

		monitor_t[i] = sorted_monitor_t[i];
		monitor_i[i] = sorted_monitor_i[i];
	}
}


int main()
{

	double t[] = {3.1,6.5,4,8,2,6};
	std::vector<double> times(t, t + sizeof(t) / sizeof(double));
	int32_t i[] = {3,0,2,4,1,2};
	std::vector<int32_t> idx(i, i + sizeof(i) / sizeof(int32_t));

	std::cout << "Original Vectors\ntimes\tidx\n" << std::endl;

	for (size_t i = 0;  i != times.size(); ++i){
		std::cout << times[i] << "\t" << idx[i] << std::endl;
	}

	sort_spikemonitor(times, idx);

	std::cout << "Sorted Vectors\ntimes\tidx\n" << std::endl;

	for (size_t i = 0;  i != times.size(); ++i){
		std::cout << times[i] << "\t" << idx[i] << std::endl;
	}
}
	

	
