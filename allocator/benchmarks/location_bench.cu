#include <location.cu>
#include <celero/Celero.h>

#include <iostream>
#include <vector>
#include <utility>

#define LOCATION_SAMPLES 20
#define LOCATION_OPERATIONS 3000
#define LOCATION_TESTS 10

/********************************test fixture******************************/
template <typename T>
class LocationFixture : public celero::TestFixture {
    public:
     typedef int64_t Size;
     typedef std::pair<Size, uint64_t> pair;

     LocationFixture() {};

     virtual std::vector<pair> getExperimentValues() const {
          std::vector<pair> problemSpace;
          const int totalNumberOfTests = LOCATION_TESTS;

          for (int i = 0; i < totalNumberOfTests; i++) {
               problemSpace.push_back(
                   std::make_pair(static_cast<Size>(std::pow(2, i + 1)), 0));
          }
          return problemSpace;
     }

     virtual void setUp(Size s) {
          size = s;
     };
     virtual void tearDown() {
     }

     Size size;
     location<T> policy_;
     void Benchmark_Function() {
          auto p = policy_.New(static_cast<std::size_t>(size));
          policy_.Delete(p);
     };
};
/********************************test fixture******************************/
/********************************Benchmarks******************************/

#define CASE(Type, name, Case)                                                                     \
     Type(LocationBenchmark, name, LocationFixture<Case>, LOCATION_SAMPLES, LOCATION_OPERATIONS) { \
          this->Benchmark_Function();                                                              \
     };

CASE(BASELINE_F, policy_HOST, host);
CASE(BENCHMARK_F, policy_DEVICE, device)
CASE(BENCHMARK_F, policy_PINNED, pinned)
CASE(BENCHMARK_F, policy_MANAGED, unified)

/********************************Benchmarks******************************/
/********************************Main Function******************************/

/********************************Main FUnction******************************/
