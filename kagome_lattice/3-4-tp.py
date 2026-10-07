import sympy as sp 

K = sp.Matrix([sp.pi/4, sp.pi/(4*sp.sqrt(3))])
# K = sp.Matrix([sp.pi/12, -sp.pi/(4*sp.sqrt(3))])

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
                             [0, ME2*dud2, dud3, 0, E2*dud2, ME3*dud3],
                             [E2*dud2, 0, 0, ME2*dud2, 0, 2*sp.cos(K.dot(delta1))*dud1],
                             [ME3*dud3, 0, 0, E3*dud3, 2*sp.cos(K.dot(delta1))*dud1, 0]])



v3 = sp.Matrix([(1-sp.sqrt(3))/2 + sp.I*(3-sp.sqrt(3))/2, sp.sqrt(2)*(1-sp.sqrt(3)), sp.I*(-2+sp.sqrt(3)), (1-sp.sqrt(3))/2 + sp.I*(sp.sqrt(3)-3)/2, 0, 1])/(2*sp.sqrt(6-3*sp.sqrt(3)))
v4 = sp.Matrix([3-2*sp.sqrt(3)+sp.I*(-2+sp.sqrt(3)), 0, sp.I*(1-sp.sqrt(3)), 3-2*sp.sqrt(3)+sp.I*(2-sp.sqrt(3)), 2*sp.sqrt(2)*(2-sp.sqrt(3)), 5-3*sp.sqrt(3)])/(2*sp.sqrt(6*(7-4*sp.sqrt(3))))

v5 = sp.Matrix([-1/sp.sqrt(2) - sp.I*(2+sp.sqrt(3))/sp.sqrt(2), -2-sp.sqrt(3), sp.I*sp.sqrt(2)*(1+sp.sqrt(3)), -1/sp.sqrt(2) +sp.I*(2+sp.sqrt(3))/sp.sqrt(2), 1, 0])/(sp.sqrt(6)+3*sp.sqrt(2))
v6 = sp.Matrix([1+sp.sqrt(3)+sp.I*(1-sp.sqrt(3)), -sp.sqrt(2)+sp.sqrt(6), 0, 1+sp.sqrt(3) -sp.I*(1-sp.sqrt(3)), sp.sqrt(2)+sp.sqrt(6), 4])/(4*sp.sqrt(3))

v4t = sp.Matrix([(1+sp.sqrt(3))/sp.sqrt(2)-sp.I*sp.sqrt(2)*(2+sp.sqrt(3)), 2+sp.sqrt(3), sp.I*sp.sqrt(2)*(2+sp.sqrt(3)), (1+sp.sqrt(3))/sp.sqrt(2) + sp.I*sp.sqrt(2)*(2+sp.sqrt(3)), 1, 0])/(sp.sqrt(6)*sp.sqrt(5*sp.sqrt(3)+9))
v5t = sp.Matrix([-sp.Rational(1, 2), -sp.sqrt(3)*(1+sp.sqrt(3))/(2*sp.sqrt(2)), sp.I*(1+sp.sqrt(3))/2, -sp.Rational(1, 2), (1+sp.sqrt(3))/(2*sp.sqrt(2)), 1])/sp.sqrt(3*sp.sqrt(3)*(1+sp.sqrt(3))/2)

k1 = v3
k2 = v4

phi = -sp.pi/2
kappa = -sp.pi/2
theta = 5*sp.pi/12

alpha = sp.cos(phi/2)*sp.exp(sp.I*(theta/2+kappa))
beta = sp.sin(phi/2)*sp.exp(sp.I*(theta/2))
eta = sp.sin(phi/2)*sp.exp(sp.I*(-theta/2+kappa))
gamma = -sp.cos(phi/2)*sp.exp(sp.I*(-theta/2))

w1 = alpha*k1 + beta*k2
w2 = eta*k1 + gamma*k2

def master_simp(a):
    return sp.simplify(sp.sqrtdenest(sp.simplify(sp.expand_complex(a))))



#print(H0.eigenvects())
# print(sp.simplify(master_simp(w1.H.dot(H1*w1))))
# print(sp.simplify(sp.expand_complex(sp.sqrtdenest(w2.H.dot(H1*w2)))))

dH111 = master_simp(w1.H.dot(dH1*w1))
redH111 = sp.factor(sp.re(dH111).expand())
imdH111 = sp.factor(sp.im(dH111).expand())


dH122 = master_simp(w2.H.dot(dH1*w2))
redH122 = sp.factor(sp.re(dH122).expand())
imdH122 = sp.factor(sp.im(dH122).expand())


#dH112 = master_simp(w1.H.dot(dH1*w2))
#redH112 = sp.factor(sp.re(dH112).expand())
#imdH112 = sp.factor(sp.im(dH112).expand())


#dH121 = master_simp(w2.H.dot(dH1*w1))
#redH121 = sp.factor(sp.re(dH121).expand())
#imdH121 = sp.factor(sp.im(dH121).expand())

print(redH111 + sp.I*imdH111)
print(redH122 + sp.I*imdH122)
#print(redH112 + sp.I*imdH112)
#print(redH121 + sp.I*imdH121)

#print(sp.simplify(sp.sqrtdenest(sp.simplify(sp.expand_complex(w2.H.dot(dH1*w2))))))
#print(sp.simplify(sp.sqrtdenest(sp.simplify(sp.expand_complex(w1.H.dot(dH1*w2))))))
#print(sp.simplify(sp.sqrtdenest(sp.simplify(sp.expand_complex(w2.H.dot(dH1*w1))))))

#print(spv4t - v4t).collect(sp.simplify(master_simp(w5.H.dot((dH0)*w5))), [px, py]))

#print(sp.collect(sp.simplify(master_simp(w6.H.dot((dH0)*w6))), [px, py]))
#print(sp.collect(sp.simplify(master_simp(w5.H.dot((dH0)*w6))), [px, py]))
#print(sp.collect(sp.simplify(sp.expand_complex(w6.H.dot((dH0)*w5))), [px, py]))


# vects = sp.GramSchmidt(H0.eigenvects()[1][2], False)


#for i in [0,1]:
    #print(vects[i].applyfunc(sp.simplify).applyfunc(master_simp))
 
