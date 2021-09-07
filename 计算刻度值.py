import math

center=[121 , 116]
a={20:( 52,189 ) ,30:( 34,168 ) ,40: ( 24,144 ),50:( 22,103 ) ,
   60:( 40,60 ) ,70:( 90,25 ) ,80:( 166,31 ) ,90:( 218,90 ) ,100:(193,186)}
count=0
result={}
for k ,v in a.items():
    r=math.acos((v[0]-center[0])/((v[0]-center[0])**2 + (v[1]-center[1])**2)**0.5)
    r=r*180/math.pi
    a[k]=r
    if count >= 4 and k != 100:
        r=360-r
        # print(k, r)
    result[k]=r
    count+=1
d=360-result[90]+result[100]
d1=360-result[90]
t=90+10*(d1/d)
result[t]=0
result_list=result.items()
lst=sorted(result_list,key=lambda x:x[1])


def get_next(c):
    l=len(lst)
    n=0
    for i in range(len(lst)):
        if lst[i][0]==c:
            n=i+1
            if n==l:
                n=0
            break
    return lst[n]


def get_rad_val(rad):
    old=None
    for k, v in lst:
        # print(k,v)
        if rad > v :
            old = k
    # print(old)
    r=result[old]
    d=rad-r
    nx=get_next(old)
    # print(10*abs(d/(nx[1] - r)))
    # print(nx)
    t=old+10*abs(d/(nx[1] - r))
    # print(t)
    return t