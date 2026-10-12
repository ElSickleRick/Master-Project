import sympy as sp 

K = sp.Matrix([2*sp.pi/3, 0]) 

delta1 = sp.Matrix([1, 0])
delta2 = sp.Matrix([sp.Rational(1,2), sp.sqrt(3)/2])
delta3 = sp.Matrix([sp.Rational(1,2), -sp.sqrt(3)/2])

Kd1 = K.dot(delta1)
Kd2 = K.dot(delta2)
Kd3 = K.dot(delta3)

px, py, p1, p2, p3 = sp.symbols('px py p1 p2 p3', real = 'True')

p1 = px
p2 = (px + sp.sqrt(3)*py)/2
p3 = (px - sp.sqrt(3)*py)/2

uxx, uxy, uyx, uyy, strainbeta = sp.symbols('uxx uxy uyx uyy strainbeta', real = 'True')
u = sp.Matrix([[uxx, uxy], [uyx, uyy]])

dud1 = delta1.dot(u*delta1)
dud2 = delta2.dot(u*delta2)
dud3 = delta3.dot(u*delta3)

Kud1 = K.dot(u*delta1)
Kud2 = K.dot(u*delta2)
Kud3 = K.dot(u*delta3)

H0 = 2*sp.Matrix([[0, sp.cos(Kd2) , sp.cos(Kd3)],
                    [sp.cos(Kd2), 0, sp.cos(Kd1)],
                    [sp.cos(Kd3), sp.cos(Kd1), 0]])
H1 = -2*sp.Matrix([[0, sp.sin(Kd2)*p2 , sp.sin(Kd3)*p3],
                    [sp.sin(Kd2)*p2, 0, sp.sin(Kd1)*p1],
                    [sp.sin(Kd3)*p3, sp.sin(Kd1)*p1, 0]])

dH0 = -2*sp.Matrix([[0, sp.pi*(uxx/sp.sqrt(3) + uyx)/2, sp.pi*(uxx/sp.sqrt(3)-uyx)/2],
                    [sp.pi*(uxx/sp.sqrt(3)+uyx)/2, 0, sp.pi*uxx/sp.sqrt(3)],
                    [sp.pi*(uxx/sp.sqrt(3)-uyx)/2, sp.pi*uxx/sp.sqrt(3), 0]])

dH1 = -2*strainbeta*sp.Matrix([[0, (uxx+2*sp.sqrt(3)*uyx + 3*uyy)/8, (uxx-2*sp.sqrt(3)*uyx+3*uyy)/8],
                    [(uxx+2*sp.sqrt(3)*uyx + 3*uyy)/8, 0, -uxx/2],
                    [(uxx-2*sp.sqrt(3)*uyx + 3*uyy)/8, -uxx/2, 0]])

v1 = sp.Matrix([1,1,0])/sp.sqrt(2)
v2 = sp.Matrix([1,-1,2])/sp.sqrt(6)


k1 = v1
k2 = v2

phi = -sp.pi/2
kappa = -sp.pi/2
theta = sp.pi/3

alpha = sp.cos(phi/2)*sp.exp(sp.I*(theta/2+kappa))
beta = sp.sin(phi/2)*sp.exp(sp.I*(theta/2))
eta = sp.sin(phi/2)*sp.exp(sp.I*(-theta/2+kappa))
gamma = -sp.cos(phi/2)*sp.exp(sp.I*(-theta/2))

w1 = alpha*k1 + beta*k2
w2 = eta*k1 + gamma*k2


def master_simp(a):
    return sp.simplify(sp.sqrtdenest(sp.simplify(sp.expand_complex(a))))


def minor_simp(a):
    return sp.simplify(sp.expand_complex(sp.simplify(sp.sqrtdenest(a))))

j1 = w1
j2 = w2

A = dH0


A11 = minor_simp(j1.H.dot(A*j1))
reA11 = sp.factor(sp.re(A11).expand())
imA11 = sp.factor(sp.im(A11).expand())

print(reA11 + sp.I*imA11)

A22 = minor_simp(j2.H.dot(A*j2))
reA22 = sp.factor(sp.re(A22).expand())
imA22 = sp.factor(sp.im(A22).expand())

print(reA22 + sp.I*imA22)

A12 = minor_simp(j1.H.dot(A*j2))
reA12 = sp.factor(sp.re(A12).expand())
imA12 = sp.factor(sp.im(A12).expand())

print(reA12 + sp.I*imA12)

A21 = minor_simp(j2.H.dot(A*j1))
reA21 = sp.factor(sp.re(A21).expand())
imA21 = sp.factor(sp.im(A21).expand())

print(reA21 + sp.I*imA21)

