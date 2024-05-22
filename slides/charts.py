from sympy import *

x, y = symbols('x y')

f = x * cos(x) * sin(y)

init_printing()

fx = Integral(f, x).doit()
fxy = Integral(fx, y).doit()

fy = Integral(f, y).doit()
fyx = Integral(fy, x).doit()

pprint(Integral(Integral(f, x), y))
print('=')
pprint(fxy)

fn = Integral(Integral(fxy, x), y)
pprint(fn)
print('=')
fn = Integral(Integral(fxy, x).doit(), y).doit()
pprint(fn)

pprint(Integral(Integral(fn, x), y))
print('=')
pprint(Integral(Integral(fn, x).doit(), y).doit())

print('----------------------------------------------------------------')

pprint(Integral(Integral(f, y), x))
print('=')
pprint(fyx)

fn = Integral(Integral(fyx, y), x)
pprint(fn)
print('=')
fn = Integral(Integral(fyx, y).doit(), x).doit()
pprint(fn)

pprint(Integral(Integral(fn, y), x))
print('=')
pprint(Integral(Integral(fn, y).doit(), x).doit())
