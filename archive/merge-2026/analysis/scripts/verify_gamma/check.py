import math, random
def gammaincc(a, x):
    # regularised upper incomplete gamma Q(a,x)
    if x <= 0: return 1.0
    if x < a+1:
        s=1.0/a; t=s; n=a
        for _ in range(100000):
            n+=1; t*=x/n; s+=t
            if abs(t)<abs(s)*1e-15: break
        return 1.0 - s*math.exp(-x+a*math.log(x)-math.lgamma(a))
    b=x+1-a; c=1e300; d=1/b; h=d
    for i in range(1,100000):
        an=-i*(i-a); b+=2; d=an*d+b; d=1/d if d!=0 else 1e300
        c=b+an/c; c=c if c!=0 else 1e-300; de=d*c; h*=de
        if abs(de-1)<1e-15: break
    return math.exp(-x+a*math.log(x)-math.lgamma(a))*h
def bisect(f,lo,hi,it=200):
    for _ in range(it):
        m=0.5*(lo+hi)
        if (f(lo)>0)==(f(m)>0): lo=m
        else: hi=m
    return 0.5*(lo+hi)
tau, eps = 1.0, 0.01
# S3: rate b=0.01 (scale 100); cap shape A_hat: Q(A, tau*b) = eps
b=0.01
Ahat=bisect(lambda A: gammaincc(A, tau*b)-eps, 1e-9, 50)
print("cap shape A_hat =", Ahat)
# schedule: mission at cap (X1 ~ Gamma(Ahat, b)), repair rho=0.5 pathwise -> 0.5*X1, then mission X2 ~ Gamma(Ahat/2, b)
# surrogate shape on day 3: 0.5*Ahat + 0.5*Ahat = Ahat -> surrogate tail = eps
random.seed(1); N=2_000_000; hit=0
for _ in range(N):
    x1=random.gammavariate(Ahat, 1/b) if Ahat>0 else 0.0
    x2=random.gammavariate(Ahat/2, 1/b)
    if 0.5*x1+x2>tau: hit+=1
p=hit/N; se=math.sqrt(p*(1-p)/N)
print(f"S3 day-3 true tail (MC) = {p:.5f} +- {se:.5f}  vs eps {eps}")
# S2 mixture: type1 mean .1 rate 10 (shape 1), type2 mean .1 rate 200 (shape 20); pair (1,7)
random.seed(2); N=1_000_000; hit=0
for _ in range(N):
    d=random.gammavariate(1*1,1/10)+random.gammavariate(20*7,1/200)
    if d>tau: hit+=1
p=hit/N; print(f"S2 pair (1,7) true tail (MC) = {p:.4f} +- {math.sqrt(p*(1-p)/N):.4f}")
# main's cap at common rate 10 and accepted surrogate shapes
Amax=bisect(lambda A: gammaincc(A, tau*10)-eps, 1e-9, 50)
print("A_max at rate 10 =", round(Amax,4), "; surrogate (1,7) =", round(1+7*0.3287,4), "; (3,2) =", round(3+2*0.3287,4))
random.seed(3); N=1_000_000; hit=0
for _ in range(N):
    d=random.gammavariate(3,1/10)+random.gammavariate(40,1/200)
    if d>tau: hit+=1
p=hit/N; print(f"S2 pair (3,2) true tail (MC) = {p:.4f} +- {math.sqrt(p*(1-p)/N):.4f}")
