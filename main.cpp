#include <symengine/symbol.h>

#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "symengine/add.h"
#include "symengine/expression.h"
#include "symengine/sets.h"
using namespace std;
using namespace SymEngine;
int main() {
  RCP<const Basic> s1 = SymEngine::symbol("s1");
  RCP<const Basic> s2 = SymEngine::symbol("s2");
  RCP<const Basic> s3 = SymEngine::symbol("s3");
  auto s4 = s2;
  auto s5 = SymEngine::add(s1, s2);
  auto s6 = SymEngine::add(s1, s4);
  std::cout<< s5->__eq__(*s6.get())<<std::endl;

  return 0;
}