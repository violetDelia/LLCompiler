#include <symengine/expression.h>
#include <symengine/symbol.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "symengine/add.h"
#include "symengine/basic.h"
#include "symengine/dict.h"
#include "symengine/expression.h"
#include "symengine/functions.h"
#include "symengine/integer.h"
#include "symengine/logic.h"
#include "symengine/mul.h"
#include "symengine/number.h"
#include "symengine/rational.h"
#include "symengine/refine.h"
#include "symengine/sets.h"
#include "symengine/test_visitors.h"
using namespace std;
using namespace SymEngine;
int main() {
  RCP<const Basic> s1 = Rational::from_two_ints(-2,100);
  const auto a1 = Assumptions({Gt(s1, integer(0))});
  std::cout <<SymEngine::ccode(*s1)<<std::endl;

  std::cout << (is_positive(*s1, &a1) == tribool::tritrue) << std::endl;
  return 0;
}