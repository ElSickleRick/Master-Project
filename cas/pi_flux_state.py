import sympy as sp 

K = sp.Matrix([sp.pi/4, sp.pi/(4*sp.sqrt(3))]) # F
# K = sp.Matrix([sp.pi/12, -sp.pi/(4*sp.sqrt(3))]) # C
# K = sp.Matrix([5*sp.pi/12, -sp.sqrt(3)*sp.pi/12]) # E

delta1 = sp.Matrix([1, 0])
delta2 = sp.Matrix([sp.Rational(1,2), sp.sqrt(3)/2])
delta3 = sp.Matrix([sp.Rational(1,2), -sp.sqrt(3)/2])

E2 = sp.exp(sp.I * K.dot(delta2))
ME2 = sp.exp(-sp.I * K.dot(delta2))

E3 = sp.exp(sp.I * K.dot(delta3))
ME3 = sp.exp(-sp.I * K.dot(delta3))

Kd1 = K.dot(delta1)

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

H0 = sp.Matrix([[0, E2, -ME3, 0, ME2, E3],
                [ME2, 0, sp.I*2*sp.sin(Kd1), E2, 0, 0],
                [-E3, -sp.I*2*sp.sin(Kd1), 0, ME3, 0, 0],
                [0, ME2, E3, 0, E2, ME3],
                [E2, 0, 0, ME2, 0, 2*sp.cos(Kd1)],
                [ME3, 0, 0, E3, 2*sp.cos(Kd1), 0]])

H1 = sp.Matrix([[0, sp.I*E2*p2, sp.I*ME3*p3, 0, -sp.I*ME2*p2, sp.I*E3*p3],
                [-sp.I*ME2*p2, 0,  sp.I*2*sp.cos(Kd1)*p1, sp.I*E2*p2, 0, 0],
                [-sp.I*E3*p3, -sp.I*2*sp.cos(Kd1)*p1, 0, -sp.I*ME3*p3, 0, 0],
                [0, -sp.I*ME2*p2, sp.I*E3*p3, 0, sp.I*E2*p2, -sp.I*ME3*p3],
                [sp.I*E2*p2, 0, 0, -sp.I*ME2*p2, 0, -2*sp.sin(Kd1)*p1],
                [-sp.I*ME3*p3, 0, 0, sp.I*E3*p3, -2*sp.sin(Kd1)*p1, 0]])

dH0 = sp.I*sp.Matrix([[0,       E2*Kud2, ME3*Kud3, 0, -ME2*Kud2, E3*Kud3],
                      [-ME2*Kud2, 0, 2*sp.cos(K.dot(delta1))*Kud1, E2*Kud2, 0, 0],
                      [-E3*Kud3,   -2*sp.cos(K.dot(delta1))*Kud1, 0, -ME3*Kud3, 0, 0],
                      [0,       -ME2*Kud2, E3*Kud3, 0, E2*Kud2, -ME3*Kud3],
                      [E2*Kud2, 0, 0, -ME2*Kud2, 0, 2*sp.I*sp.sin(K.dot(delta1))*Kud1],
                      [-ME3*Kud3,   0, 0, E3*Kud3, 2*sp.I*sp.sin(K.dot(delta1))*Kud1, 0]])

dH1 = -strainbeta*sp.Matrix([[0, E2*dud2, -ME3*dud3, 0, ME2*dud2, E3*dud3],
                             [ME2*dud2, 0, sp.I*2*sp.sin(K.dot(delta1))*dud1, E2*dud2, 0, 0],
                             [-E3*dud3, -sp.I*2*sp.sin(K.dot(delta1))*dud1, 0, ME3*dud3, 0, 0],
                             [0, ME2*dud2, E3*dud3, 0, E2*dud2, ME3*dud3],
                             [E2*dud2, 0, 0, ME2*dud2, 0, 2*sp.cos(K.dot(delta1))*dud1],
                             [ME3*dud3, 0, 0, E3*dud3, 2*sp.cos(K.dot(delta1))*dud1, 0]])



v3 = sp.Matrix([(1-sp.sqrt(3))/2 + sp.I*(3-sp.sqrt(3))/2, sp.sqrt(2)*(1-sp.sqrt(3)), sp.I*(-2+sp.sqrt(3)), (1-sp.sqrt(3))/2 + sp.I*(sp.sqrt(3)-3)/2, 0, 1])/(2*sp.sqrt(6-3*sp.sqrt(3)))
v4 = sp.Matrix([3-2*sp.sqrt(3)+sp.I*(-2+sp.sqrt(3)), 0, sp.I*(1-sp.sqrt(3)), 3-2*sp.sqrt(3)+sp.I*(2-sp.sqrt(3)), 2*sp.sqrt(2)*(2-sp.sqrt(3)), 5-3*sp.sqrt(3)])/(2*sp.sqrt(6*(7-4*sp.sqrt(3))))

v5 = sp.Matrix([-1/sp.sqrt(2) - sp.I*(2+sp.sqrt(3))/sp.sqrt(2), -2-sp.sqrt(3), sp.I*sp.sqrt(2)*(1+sp.sqrt(3)), -1/sp.sqrt(2) +sp.I*(2+sp.sqrt(3))/sp.sqrt(2), 1, 0])/(sp.sqrt(6)+3*sp.sqrt(2))
v6 = sp.Matrix([1+sp.sqrt(3)+sp.I*(1-sp.sqrt(3)), -sp.sqrt(2)+sp.sqrt(6), 0, 1+sp.sqrt(3) -sp.I*(1-sp.sqrt(3)), sp.sqrt(2)+sp.sqrt(6), 4])/(4*sp.sqrt(3))

v4c = sp.Matrix([(1+sp.sqrt(3))/sp.sqrt(2)-sp.I*sp.sqrt(2)*(2+sp.sqrt(3)), 2+sp.sqrt(3), sp.I*sp.sqrt(2)*(2+sp.sqrt(3)), (1+sp.sqrt(3))/sp.sqrt(2) + sp.I*sp.sqrt(2)*(2+sp.sqrt(3)), 1, 0])/(sp.sqrt(6)*sp.sqrt(5*sp.sqrt(3)+9))
v5c = sp.Matrix([-sp.Rational(1, 2), -sp.sqrt(3)*(1+sp.sqrt(3))/(2*sp.sqrt(2)), sp.I*(1+sp.sqrt(3))/2, -sp.Rational(1, 2), (1+sp.sqrt(3))/(2*sp.sqrt(2)), 1])/sp.sqrt(3*sp.sqrt(3)*(1+sp.sqrt(3))/2)

v4elegacy = sp.Matrix([(1-sp.sqrt(3))/sp.sqrt(2) -sp.I*sp.sqrt(2)*(2-sp.sqrt(3)), 2-sp.sqrt(3), sp.I*sp.sqrt(2)*(sp.sqrt(3)-2), (1-sp.sqrt(3))/sp.sqrt(2)+sp.I*sp.sqrt(2)*(2-sp.sqrt(3)), 1, 0])/sp.sqrt(6*(9-5*sp.sqrt(3)))
v5elegacy = sp.Matrix([sp.Rational(1,2), -sp.sqrt(3)*(1-sp.sqrt(3))/(2*sp.sqrt(2)), sp.I*(1-sp.sqrt(3))/2, sp.Rational(1,2), -(1-sp.sqrt(3))/(2*sp.sqrt(2)), 1])/sp.sqrt((3*(3-sp.sqrt(3)))/2)


v4e = sp.Matrix([(2-sp.sqrt(3))*sp.sqrt(2)+sp.I*(1-sp.sqrt(3))/sp.sqrt(2), -2+sp.sqrt(3), sp.I*sp.sqrt(2)*(2-sp.sqrt(3)), sp.sqrt(2)*(2-sp.sqrt(3)) -sp.I*(1-sp.sqrt(3))/sp.sqrt(2), 1, 0])/sp.sqrt(6*(9-5*sp.sqrt(3)))
v5e = sp.Matrix([sp.I/2, sp.sqrt(3)*(1-sp.sqrt(3))/(2*sp.sqrt(2)), -sp.I*(1-sp.sqrt(3))/2, -sp.I/2, -(1-sp.sqrt(3))/(2*sp.sqrt(2)), 1])/sp.sqrt(3*(3-sp.sqrt(3))/2)

k1 = v5
k2 = v6

phi = sp.pi/2
kappa = -sp.pi/2
theta = sp.pi/12

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

A = dH1


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




