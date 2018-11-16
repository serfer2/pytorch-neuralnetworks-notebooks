

def calc(prev1, prev2, prev3, rate):
    return (prev1 - rate) + (prev2 - rate) - (prev3 - rate)

a = 3.1
b = 4.1
c = 9.9
r = 0.1
n = 1

while (a + b - c) < 0:
    print('[{n}]=>({a} + {b} - {c}) = {res}'.format(n=n, a=a, b=b, c=c, res=(a + b - c)))
    n += 1
    a += r
    b += r
    c -= r

print('[{n}]=>({a} + {b} - {c}) = {res}'.format(n=n, a=a, b=b, c=c, res=(a + b - c)))