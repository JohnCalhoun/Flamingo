#include <buddy_alloc_p.cpp>
#include <standard_alloc_p.cpp>

#include <location.cu>
#include <celero/Celero.h>

#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include <random>

#include <MacroUtilities.cpp>

#define ALLOCATION_TEST 8
#define ALLOCATION_THREADS 8

// ********************************test fixture****************************** /

template <typename T>
class AllocationFixture : public celero::TestFixture {
    public:
     typedef typename T::size_type Type;
     typedef typename T::pointer pointer;
     typedef int64_t Problem_Value;

     typedef std::pair<Problem_Value, uint64_t> pair;

     virtual std::vector<pair> getExperimentValues() const {
          std::vector<pair> problemSpace;

          for (int i = 0; i < ALLOCATION_TEST; i++) {
               problemSpace.push_back(
                   std::make_pair(static_cast<Problem_Value>(std::pow(2, i + 1)), 0));
          }
          return problemSpace;
     }

     virtual void setUp(Problem_Value p) {
          problem_value_ = p;
          std::random_device rd;
          std::mt19937 gen(rd());
          std::uniform_int_distribution<> dist(2, 100);

          for (int i = 0; i < p; i++) {
               random_vector.push_back(dist(gen));
          }
     };
     virtual void tearDown() {
     }

     std::vector<int> random_vector;
     std::size_t unit_size = sizeof(Type);
     Problem_Value problem_value_;
     T allocator_;
     // benchmark functions
     DEFINE(SingleAllocation, ALLOCATION_THREADS)
     DEFINE(MultipleAllocation, ALLOCATION_THREADS)
     DEFINE(RandomAllocation, ALLOCATION_THREADS)
};

template <typename T>
void AllocationFixture<T>::SingleAllocation() {
     auto p = allocator_.allocate(unit_size * problem_value_);
     asm("");
     allocator_.deallocate(p);
}
template <typename T>
void AllocationFixture<T>::MultipleAllocation() {
     std::vector<pointer> vect;

     std::for_each(random_vector.begin(), random_vector.end(),
                   [this, &vect](int size) mutable {
    pointer p = allocator_.allocate(unit_size);
    asm("");
    vect.push_back(p);
     });

     std::for_each(vect.begin(), vect.end(),
                   [this](pointer p) mutable { allocator_.deallocate(p); });
}
template <typename T>
void AllocationFixture<T>::RandomAllocation() {
     std::vector<pointer> vect;

     std::for_each(random_vector.begin(), random_vector.end(),
                   [this, &vect](int size) mutable {
    pointer p = allocator_.allocate(size);
    asm("");
    vect.push_back(p);
     });

     std::for_each(vect.begin(), vect.end(),
                   [this](pointer p) mutable { allocator_.deallocate(p); });
}

// ********************************test fixture****************************** /
// ********************************Benchmarks****************************** /

#define BUDDY buddy_alloc_policy
#define STANDARD standard_alloc_policy

#define ALLOCATION_SAMPLES 15
#define ALLOCATION_OPERATIONS 30

#define HOST host
#define PINNED pinned
#define DEVICE device
#define MANAGED unified
// clang-format off
// declare type,function,location list for python script
// python:key:functions=SingleAllocation MultipleAllocation RandomAllocation
// python:key:concurency=Single Threaded
// python:key:types=int double
// python:key:locations=HOST PINNED DEVICE MANAGED

// python:template=BASELINE_F(allocator_|functions|_|concurency|_|types|_|locations|,policy_STANDARD__function_|functions|__concurency_|concurency|__datatype_|types|__location_|locations|,$AllocationFixture<STANDARD<|types|,location<|locations|> > >$,ALLOCATION_SAMPLES,ALLOCATION_OPERATIONS){this->|functions||concurency|();}

// python:template=BENCHMARK_F(allocator_|functions|_|concurency|_|types|_|locations|,policy_BUDDY__function_|functions|__concurency_|concurency|__datatype_|types|__location_|locations|,$AllocationFixture<BUDDY<|types|,location<|locations|> > >$,ALLOCATION_SAMPLES,ALLOCATION_OPERATIONS){this->|functions||concurency|();}

// python:start
// python:include=allocation.bench
#include \
    "allocation.bench"
// python:end
// clang-format on
#undef BUDDY
#undef STANDARD

#undef HOST
#undef PINNED
#undef DEVICE
#undef MANAGED
