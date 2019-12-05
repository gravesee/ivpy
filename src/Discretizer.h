#include <Rcpp.h>
#include <queue>
using namespace Rcpp;

//#define DEBUG

/***
 * Discretizer abstract class
 * Expected Inputs:
 *  @param NumericVector x - continuous variable to discretize
 *  @param NumericVector y - target variable to guide discretization
 *  @param NumericVector w - continuous, positive weight variable
 *  @param List options - List of options to pass on to derived class
 *
 * All derived classes must implement the discretize and find_best_split methods.
 * Discretize contains the iterative logic to parition the x-vector into bins.
 * While find_best_split returns a bool for whether a best split was found (
 * according to options criteria) and modifies the best_split_index argument
 * in-place.
 *
 * Discretize must return a NumericVector of split values that R can use to
 * cut the x variable.
 *
 */
class Discretizer {

public:
  Discretizer(NumericVector, NumericVector, NumericVector, List);
  virtual NumericVector discretize() = 0;
  NumericMatrix get_data();

protected:
  Rcpp::NumericMatrix data;
  Rcpp::List options;
  std::queue<std::pair<size_t, size_t> > queue; // loaded with intervals to search
  int num_bins;
  virtual bool find_best_split(std::pair<size_t, size_t>, size_t*) = 0;

};

bool comparator ( const std::pair<int, double>& l, const std::pair<int, double>& r)
{ return l.second < r.second; }

// constructor for abstract class
/*
 * @param x NumericVector - continuous variable to discretize
 * @param y NumericVector - target variable to guide supervised discretization
 * @param w NumericVector - weight variable
 * @param options List - named list of options that must be checked in the derived class
 */
Discretizer::Discretizer(NumericVector x, NumericVector y, NumericVector w, List options) :
  options(options) {

  num_bins = 0;

  // check that vectors are the same size
  if ((x.size() != y.size()) &&  (x.size()!= w.size())) {
    Rcpp::stop("x, y, and w are not the same size");
  }

  // load x into a vector of standard pairs to sort the indices
  std::vector<std::pair<int, double> > idx;
  for (int i = 0; i < x.size(); ++i) {
    idx.push_back(std::make_pair(i, x[i]));
  }
  std::sort(idx.begin(), idx.end(), comparator);

  // load the matrix in sorted order
  data = Rcpp::NumericMatrix(x.size(), 3);
  for (size_t i = 0; i < idx.size(); ++i) {
    data(i, 0) = x[idx[i].first];
    data(i, 1) = y[idx[i].first];
    data(i, 2) = w[idx[i].first];
  }

#ifdef DEBUG
  Rprintf("Exiting Constructor");
#endif

};

NumericMatrix Discretizer::get_data() {
  return Rcpp::wrap(data);
}
