import symengine as sym
from symengine import Integer, I, S, Symbol, pi, Rational
from symengine import cse, sqrt, symbols

from symengine.lib.symengine_wrapper import (true, false, Eq, Ne, Ge, Gt, Le, Lt, Symbol,
                                            I, And, Or, Not, Nand, Nor, Xor, Xnor, Piecewise,
                                            Contains, Interval, FiniteSet, oo, log,perfect_power, is_square, integer_nthroot)
a, b = sym.symbols("a,b")
c = b-a
s = sym.symbols("c100")
d =a -2
e =b -2
f = e-d
s1 = (d+2) *(e+2)
s2 = a*b
print(c.expand())
print(f.expand())
print(c.expand() == f.expand())
print(Le(d.expand(), a.expand()) == False)
print(s1.expand())
print(s2.expand())
print(s1.expand() == s2.expand())
