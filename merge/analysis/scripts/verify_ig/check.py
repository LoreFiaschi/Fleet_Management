import math
def Phi(x): return 0.5*math.erfc(-x/math.sqrt(2))
def ig_sf(x, mu, lam):
    # survival P(X > x) for IG(mu, lam)
    a = math.sqrt(lam/x)*(x/mu-1.0); b = math.sqrt(lam/x)*(x/mu+1.0)
    # term2 = exp(2lam/mu)*Phi(-b), computed in log space
    t2 = math.exp(2*lam/mu + math.log(max(Phi(-b),1e-300)))
    return 1.0 - (Phi(a) + t2)
def bisect(f, lo, hi, it=200):
    for _ in range(it):
        m=0.5*(lo+hi)
        if f(lo)*f(m) <= 0: hi=m
        else: lo=m
    return 0.5*(lo+hi)
tau=1.0
# Model cap (lore): IG(mu, mu/eta), P(D>tau)=eps
def model_cap(eta, eps):
    return bisect(lambda m: ig_sf(tau, m, m/eta)-eps, 1e-6, tau*(1-1e-9))
for eta, eps, n in [(10,0.1,3),(100,0.01,2),(1,0.01,3),(0.5,0.01,4)]:
    mb = model_cap(eta, eps)
    a = mb/n
    # n equal independent increments IG(a, a/eta): sum is IG(n a, n^2 a/eta) exactly
    true_tail = ig_sf(tau, n*a, (n*n)*a/eta)
    print(f"eta={eta} eps={eps} n={n}: model cap mu_bar={mb:.4f}, true tail={true_tail:.5f}, ratio={true_tail/eps:.3f}")
# CVaR cap: mu_hat = tau / E[Z | Z > q], Z ~ IG(1, 1/eta); numeric integration
def ig_pdf(z, mu, lam):
    return math.sqrt(lam/(2*math.pi*z**3))*math.exp(-lam*(z-mu)**2/(2*mu*mu*z))
def cvar(eta, eps):
    q = bisect(lambda z: ig_sf(z,1.0,1.0/eta)-eps, 1e-9, 1e4)
    # E[Z 1{Z>q}] by Simpson on a log grid up to a far upper limit
    lo, hi, N = math.log(q), math.log(q*1e3+50*eta+50), 20000
    h=(hi-lo)/N; s=0.0
    for k in range(N+1):
        u=lo+k*h; z=math.exp(u); w=1 if k in (0,N) else (4 if k%2 else 2)
        s+=w*z*ig_pdf(z,1.0,1.0/eta)*z   # dz = z du
    tail_mean = s*h/3
    return tail_mean/eps, q
for eta in [0.1, 0.5, 1.0]:
    eps=0.01
    c,q = cvar(eta,eps); mhat = tau/c; mb = model_cap(eta,eps)
    print(f"CVaR cap eta={eta} eps={eps}: q={q:.4f} CVaR={c:.4f} mu_hat={mhat:.4f} lore mu_bar={mb:.4f} cost={(1-mhat/mb)*100:.1f}%")
