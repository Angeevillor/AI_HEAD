def gcd(a, b):
    if b==0:
        return abs(a)
    else:
        return gcd(b,a%b)
def gcd_integer_solution(p, q):
    if q == 0:
        return 1, 0
    else:
        x, y = gcd_integer_solution(q, p%q)
        x, y = y, x - (p//q)*y
        return [x,y]
def K_calculate(a,b):
    if abs(a)==abs(gcd(a,b)):
        return [1,0]
    elif abs(b)==abs(gcd(a,b)):
        return [0,1]
    else:
        if abs(a)<=abs(b):
            return gcd_integer_solution(a,b)
        else:
            [x,y]=gcd_integer_solution(b,a)
            return [y,x]