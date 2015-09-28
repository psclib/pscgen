/*
 * sortindex.hpp
 *
 *  Created on: Oct 29, 2014
 *      Author: Au
 */

#ifndef SORTINDEX_HPP_
#define SORTINDEX_HPP_

#include <vector>

using namespace std;

template<typename T> class CompareIndicesByAnotherVectorValues {
	std::vector<T> _values;
public:
	CompareIndicesByAnotherVectorValues(const std::vector<T>& values) :
			_values(values) {
	}
public:
	bool operator()(const int& a, const int& b) const {
		return _values[a] < _values[b];
	}
};
template<typename T> class CompareIndicesByAnotherVectorValuesMax {
	std::vector<T> _values;
public:
	CompareIndicesByAnotherVectorValuesMax(const std::vector<T>& values) :
			_values(values) {
	}
public:
	bool operator()(const int& a, const int& b) const {
		return _values[a] > _values[b];
	}
};
template<typename T>
vector<size_t> sort_indexes(const vector<T> &v) {
	// initialize original index locations
	vector<size_t> idx(v.size());
	for (size_t i = 0; i != idx.size(); ++i)
		idx[i] = i;

	// sort indexes based on comparing values in v
	sort(idx.begin(), idx.end(), CompareIndicesByAnotherVectorValues<T>(v));

	return idx;
}
template<typename T>
vector<size_t> sort_indexes_max(const vector<T> &v) {
	// initialize original index locations
	vector<size_t> idx(v.size());
	for (size_t i = 0; i != idx.size(); ++i)
		idx[i] = i;

	// sort indexes based on comparing values in v
	sort(idx.begin(), idx.end(), CompareIndicesByAnotherVectorValuesMax<T>(v));

	return idx;
}

#endif /* SORTINDEX_HPP_ */
